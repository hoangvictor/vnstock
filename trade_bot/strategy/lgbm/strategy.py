from datetime import timedelta
from trade_bot.strategy.common import Strategy as CommonStrategy
from trade_bot.strategy.lgbm.features_engineering import create_features
from trade_bot.strategy.lgbm.train import train_model


class LGBMStrategy(CommonStrategy):
    def __init__(
        self,
        retrain_period: timedelta,
        valid_period: timedelta = timedelta(days=14),
        test_period: timedelta = timedelta(days=14),
    ):
        self.best_model = None
        self.start_time = None
        self.last_retrain_time = None
        self.features_data = None
        self.retrain_period = retrain_period
        self.valid_period = valid_period
        self.test_period = test_period

    def act(self, data, current_time):
        if self.start_time is None:
            print(f'Setting start time as {current_time} in LGBMStrategy')
            self.set_start_simulation_time(current_time)

        if len(data) > 0:
            symbols = list(data.keys())
            self.features_data = create_features(data=data, features_data=self.features_data, from_scratch=current_time==self.start_time)
            if current_time == self.start_time or current_time - self.last_retrain_time >= self.retrain_period:
                # Train the model
                self.best_model = train_model(features_data=self.features_data, valid_period=self.valid_period, test_period=self.test_period, current_time=current_time)
                self.last_retrain_time = current_time
            # Make predictions
            return []
        else:
            return []
        
    def set_start_simulation_time(self, start_time):
        self.start_time = start_time
        self.last_retrain_time = start_time