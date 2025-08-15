import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import timedelta
from trade_bot.account_manager import AccountManager, Order
from trade_bot.strategy.common import Strategy as CommonStrategy
from trade_bot.strategy.lgbm.features_engineering import create_features, get_current_features
from trade_bot.strategy.lgbm.train import train_model


class LGBMStrategy(CommonStrategy):
    def __init__(
        self,
        account_manager: AccountManager,
        retrain_period: timedelta,
        test_period: timedelta = timedelta(days=14)
    ):
        self.max_orders = 4
        self.count_threshold = 2
        self.num_top_symbols = 10
        self.prediction_threshold = 0.5
        self.top_symbols_scores = {}
        self.top_symbols_scores_count = {}
        self.account_manager = account_manager
        self.best_model = None
        self.start_time = None
        self.last_retrain_time = None
        self.features_data = None
        self.retrain_period = retrain_period
        self.test_period = test_period

    def act(self, data, current_time):
        if self.start_time is None:
            print(f'Setting start time as {current_time} in LGBMStrategy')
            self.set_start_simulation_time(current_time)

        if len(data) > 0:
            active_orders = self.account_manager.get_active_orders()
            active_symbols = [order.symbol for order in active_orders]
            money_balance = self.account_manager.get_money_balance()
            all_balance = self.account_manager.get_all_balance()

            for order in active_orders:
                if order.status == 'no_price_data':
                    order.buy_time += timedelta(days=1)
                    order.take_profit_time += timedelta(days=1)
          
            self.features_data = create_features(data=data, features_data=self.features_data, from_scratch=current_time==self.start_time)
            if current_time == self.start_time or current_time - self.last_retrain_time >= self.retrain_period:
                # Train the model
                model_data = train_model(features_data=self.features_data, test_period=self.test_period, current_time=current_time, label_deltatime=timedelta(days=10))
                self.best_model = model_data['model']
                self.best_threshold = model_data['threshold']
                self.top_threshold = model_data['top_threshold']
                self.last_retrain_time = current_time
            
            active_orders_length = len(active_orders)
            current_features, org_df = get_current_features(features_data=self.features_data, current_time=current_time)
            current_features['symbol'] = org_df['symbol']
            current_features.to_csv(f'./run_data/current_features_{current_time}.csv', index=False)

            for order in active_orders:
                if order.status == 'triggered' and order.direction == 'buy':
                    symbol = order.symbol
                    volume = order.volume
                    buy_price = order.buy_price
                    if data[symbol].shape[0] == 0:
                        continue
                    
                    if current_time - timedelta(days=2) >= order.buy_time:
                        close_price = data[symbol].iloc[-1]['close']
                        if close_price * (1 + order.cutloss_rate) < buy_price:
                            order.take_profit_time = current_time
                            order.type = 'open'
                            order.note = 'cutloss'
                            continue

                        # Check for sell signals
                        if close_price > order.buy_price * (1 + order.take_profit_rate):
                            order.take_profit_rate = (close_price / order.buy_price) - 1.03
                            order.trigger_take_profit = True
                            order.note = 'change take profit rate'
                            order.take_profit_time = min(order.take_profit_time + timedelta(days=10), order.buy_time + timedelta(days=20))
                            continue

                        if order.trigger_take_profit and close_price < order.buy_price * (1 + order.take_profit_rate):
                            order.note = f'sell for profit {order.take_profit_rate}'
                            order.take_profit_time = current_time
                            order.type = 'open'
                        
                        # sell_signals_df = current_features[(current_features['cur_volume_vs_avg10_diff'] < -0.4) & (current_features['prev1d_cur_volume_vs_avg10_diff'] < -0.4) & (current_features['prev2d_cur_volume_vs_avg10_diff'] < -0.4)]
                        # if sell_signals_df[sell_signals_df['symbol'] == symbol].shape[0] > 0:
                        #     order.note = 'sell because of volume signals'
                        #     order.take_profit_time = current_time
                        #     order.type = 'open'

            del current_features['symbol']
            
            if current_features.shape[0] == 0:
                return []
        
            predictions = self.best_model.predict(current_features)
            org_df.loc[:, 'prediction'] = predictions
            org_df['close_5d_vs_10d_pct_change'] = current_features['close_5d_vs_10d_pct_change']
            org_df['close_10d_pct_change'] = current_features['close_10d_pct_change']
            org_df['close_5d_pct_change'] = current_features['close_5d_pct_change']
            org_df['close_3d_pct_change'] = current_features['close_3d_pct_change']
            org_df['close_1d_pct_change'] = current_features['close_1d_pct_change']
            org_df['close_slope10'] = current_features['close_slope10']
            org_df['cur_close_vs_prev_max_high_5d_diff'] = current_features['cur_close_vs_prev_max_high_5d_diff']
            org_df['close_open_diff'] = current_features['close_open_diff']
            org_df['close_slope40'] = current_features['close_slope40']

            org_df.loc[:, 'buy'] = False
            # if self.best_threshold is not None:
            org_df.loc[:, 'buy'] = (org_df['prediction'] > self.prediction_threshold) # & (org_df['close_slope40'] > 0.) & (org_df['close_slope40'] < 0.3) & (org_df['close_slope10'] < 0.05) & (org_df['close_slope10'] > 0.)
            # (org_df['prediction'] > self.prediction_threshold) & (org_df['close_slope10'].abs() < 0.3) & (org_df['close_slope10'] > 0.05) & (org_df['close_slope40'] < 0.6) & (org_df['close_open_diff'] > -0.05)
            
            filename = f'./run_data/org_df_{current_time}.csv'
            org_df = org_df.sort_values(by='prediction', ascending=False).reset_index(drop=True)
            org_df.to_csv(filename, index=False)
            
            buy_df = org_df[org_df['buy']]
            buy_df = buy_df.sort_values(by='prediction', ascending=False)

            for symbol in self.top_symbols_scores.keys():
                if symbol not in buy_df['symbol'].values:
                    self.top_symbols_scores[symbol] = self.top_symbols_scores[symbol] * 0.5
            
            for _, row in buy_df.iterrows():
                symbol = row['symbol']
                if symbol in self.top_symbols_scores:
                    self.top_symbols_scores[symbol] = (self.top_symbols_scores[symbol] + row['prediction'])/2
                else:
                    self.top_symbols_scores[symbol] = row['prediction']
                self.top_symbols_scores_count[symbol] = self.top_symbols_scores_count.get(symbol, 0) + 1

                if len(self.top_symbols_scores) > self.num_top_symbols:
                    break
        
            all_symbols = list(self.top_symbols_scores.keys())
            for symbol in all_symbols:
                if self.top_symbols_scores[symbol] < self.prediction_threshold:
                    del self.top_symbols_scores[symbol]
            
            orders = []
            cur_order_count = 0
            top_symbols_scores_count_df = pd.DataFrame(self.top_symbols_scores_count.items(), columns=['symbol', 'count'])
            top_symbols_scores_df = pd.DataFrame(self.top_symbols_scores.items(), columns=['symbol', 'score'])
            top_symbols_scores_df = top_symbols_scores_df.merge(org_df[['symbol', 'close']], on='symbol', how='left')
            top_symbols_scores_df = top_symbols_scores_df.merge(top_symbols_scores_count_df, on='symbol', how='left')
            top_symbols_scores_df = top_symbols_scores_df.sort_values(by='score', ascending=False)
            top_symbols_scores_df.to_csv(f'./run_data/top_symbols_scores_df_{current_time}.csv', index=False)
            
            if active_orders_length > self.max_orders or money_balance < all_balance/self.max_orders:
                return []

            for _, row in top_symbols_scores_df.iterrows():
                symbol = row['symbol']
                count = row['count']
                if count < self.count_threshold or bool(np.isnan(row['close'])):
                    continue
                volume = float(all_balance/self.max_orders/row['close'])
                volume -= volume % 100
                if volume == 0:
                    continue
                if symbol in active_symbols:
                    continue
                orders.append(Order(
                    symbol = symbol,
                    volume = volume,
                    direction = 'buy',
                    type = 'open',
                    buy_time = current_time + timedelta(days=1),
                    take_profit_time = current_time + timedelta(days=10))
                )
                cur_order_count += 1
                active_orders_length += 1
                if active_orders_length >= 4:
                    break
            return orders
        else:
            return []
        
    def set_start_simulation_time(self, start_time):
        self.start_time = start_time
        self.last_retrain_time = start_time