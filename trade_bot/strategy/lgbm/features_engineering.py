import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import linregress

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from talipp.ohlcv import OHLCVFactory
from talipp.indicators import ADX, AO, Aroon, ATR, BOP, CHOP, CoppockCurve, DPO, EMV, IBS, \
    MACD, MassIndex, ROC, RSI, STC, Stoch, StochRSI, SuperTrend, TRIX, TSI, TTM, UO, VTX


USE_CACHE: bool = os.getenv('USE_CACHE', '0') == '1'
CACHE_DIR: str = os.getenv('CACHE_DIR', './cache')
os.makedirs(CACHE_DIR, exist_ok=True)

class Features:
    def __init__(self, symbol: str, org_df: pd.DataFrame, predicted_days: int=10):
        self.symbol = symbol
        self.org_df = org_df
        self.predicted_days = predicted_days
        self.features_df = None
    
    @staticmethod
    def get_volume_level(x):
        if x < 1e7:
            return 0
        if x < 5e7:
            return 1
        if x < 1e8:
            return 2
        return 3

    @staticmethod
    def estimate_increasing_rate(x):
        d = len(x)
        lr = linregress(range(d), x/x.tolist()[-1])
        i = lr.intercept.tolist()
        s = lr.slope.tolist()
        if i == 0:
            return s*d
        return s*d/i
    
    def create_basic_ohlcv_features(self, df):
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        df['close_open_diff'] = (df['close'] - df['open']) / df['open']
        df['high_low_diff'] = (df['high'] - df['low']) / df['low']
        df['close_low_diff'] = (df['close'] - df['low']) / df['low']
        df['volume_level'] = (df['volume'] * df['close']).apply(self.get_volume_level)
        df['volume_level'] = df['volume_level'].astype("category")

        # Percentage change
        for col in ['close', 'volume', 'high', 'low']:
            for d in [1, 3, 5, 10, 20, 40]:
                new_col = f'{col}_{d}d_pct_change'
                df[new_col] = df[col].pct_change(d)
                df.loc[df[new_col] == np.inf, new_col] = 1
                df.loc[df[new_col] == -np.inf, new_col] = -1

            for date_suffix in ['3d_vs_5d_pct_change', '5d_vs_10d_pct_change', '10d_vs_20d_pct_change']:
                new_col = f'{col}_{date_suffix}'
                first_date = int(date_suffix.split('_')[0][:-1])
                second_date = int(date_suffix.split('_')[2][:-1])
                df[new_col] = df[col].shift(first_date)/df[col].shift(second_date)-1
                df.loc[df[new_col] == np.inf, new_col] = 1
                df.loc[df[new_col] == -np.inf, new_col] = -1

        # Moving average
        for col in ['close', 'volume', 'high', 'low']:
            for d in [5, 10, 20, 40]:
                df[f'{col}_avg{d}'] = df[col].rolling(window=d).mean()
                df[f'cur_{col}_vs_avg{d}_diff'] = (df[col] - df[f'{col}_avg{d}']) / df[f'{col}_avg{d}']
                del df[f'{col}_avg{d}']

        for col in ['volume', 'close', 'high', 'low']:
            for d in [5, 10, 20, 40]:
                df[f'{col}_slope{d}'] = df[col].rolling(d).apply(self.estimate_increasing_rate)
                df.loc[df[f'{col}_slope{d}'].isna(), f'{col}_slope{d}'] = 0

        for col in ['volume', 'close', 'high', 'low']:
            df[f'{col}_slope5_vs_slope10_diff'] = (1+df[f'{col}_slope5']) / (1+df[f'{col}_slope10'])-1
            df[f'{col}_slope5_vs_slope20_diff'] = (1+df[f'{col}_slope5']) / (1+df[f'{col}_slope20'])-1
            df[f'{col}_slope10_vs_slope20_diff'] = (1+df[f'{col}_slope10']) / (1+df[f'{col}_slope20'])-1
            df[f'{col}_slope10_vs_slope40_diff'] = (1+df[f'{col}_slope10']) / (1+df[f'{col}_slope40'])-1
            df[f'{col}_slope20_vs_slope40_diff'] = (1+df[f'{col}_slope20']) / (1+df[f'{col}_slope40'])-1

        return df

    def create_shifted_features(self, df):
        shift_cols = [c for c in df.columns if not c.startswith('prev') and c not in ['time', 'open', 'high', 'low', 'close', 'volume', 'next_close', 'next_min_close', 'label', 'weight', 'symbol']]
        for shift_day in range(1,4):
            df[[f'prev{shift_day}d_{sc}' for sc in shift_cols]] = df.shift(shift_day)[shift_cols]
        return df

    def create_candle_related_features(self, df):
        df['prev_max_high_5d'] = df['high'].rolling(window=5).max().shift(1)
        df['prev_min_low_5d'] = df['low'].rolling(window=5).min().shift(1)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)

        for price_type in ['close', 'open', 'high', 'low']:
            df[f'cur_{price_type}_vs_prev_max_high_5d_diff'] = (df[price_type] - df['prev_max_high_5d']) / df['prev_max_high_5d']
            df[f'cur_{price_type}_vs_prev_min_low_5d_diff'] = (df[price_type] - df['prev_min_low_5d']) / df['prev_min_low_5d']
            df[f'cur_{price_type}_vs_prev_max_high_diff'] = (df[price_type] - df['prev_high']) / df['prev_high']
            df[f'cur_{price_type}_vs_prev_min_low_diff'] = (df[price_type] - df['prev_low']) / df['prev_low']

        for n in [3, 5, 10, 20]:
            df[f'incr_candles_count_{n}d'] = (df['close'] > df['open']).rolling(window=n).sum()
            df[f'decr_candles_count_{n}d'] = (df['close'] < df['open']).rolling(window=n).sum()
            # Sum of absolute differences for increasing candles
            df[f'incr_candles_abs_diff_sum_{n}d'] = (
                (df['close'] / df['open'] - 1).where(df['close'] > df['open'], 0).rolling(window=n).sum()
            )
            
            # Sum of absolute differences for decreasing candles
            df[f'decr_candles_abs_diff_sum_{n}d'] = (
                (df['open'] / df['close'] - 1).where(df['close'] < df['open'], 0).rolling(window=n).sum()
            )

        del df['prev_max_high_5d'], df['prev_min_low_5d'], df['prev_high'], df['prev_low']
        return df

    def create_features_from_scratch(self):

        df = self.org_df.copy()
        df = self.create_basic_ohlcv_features(df)
        df = self.create_candle_related_features(df)
        ohlcv = OHLCVFactory.from_dict({
            "open": df['open'],
            "high": df['high'],
            "low": df['low'],
            "close": df['close'],
            "volume": df['volume']
        })
        close = df['close']
        self.adx = ADX(10, 10, ohlcv)
        self.ao = AO(5, 34, ohlcv)
        self.aroon = Aroon(10, ohlcv)
        # self.atr = ATR(10, ohlcv)
        self.bop = BOP(ohlcv)
        self.chop = CHOP(10, ohlcv)
        self.coppock_curve = CoppockCurve(11, 10, 10, close)
        self.dpo = DPO(20, close)
        self.emv = EMV(10, 10000, ohlcv)
        self.ibs = IBS(ohlcv)
        self.macd = MACD(12, 26, 9, close)
        self.mass_index = MassIndex(9, 9, 10, ohlcv)
        self.roc = ROC(9, close)
        self.rsi = RSI(10, close)
        self.stc = STC(23, 50, 10, 3, close)
        self.stoch = Stoch(10, 3, ohlcv)
        self.stoch_rsi = StochRSI(10, 10, 3, 3, close)
        # self.super_trend = SuperTrend(10, 3, ohlcv)
        # self.trix = TRIX(18, close)
        self.tsi = TSI(13, 25, close)
        self.ttm = TTM(20, input_values = ohlcv)
        self.uo = UO(7, 10, 20, ohlcv)
        self.vtx = VTX(10, ohlcv)

        ao = self.ao.output_values
        # atr = self.atr.output_values
        bop = self.bop.output_values
        chop = self.chop.output_values
        coppock_curve = self.coppock_curve.output_values
        dpo = self.dpo.output_values
        emv = self.emv.output_values
        ibs = self.ibs.output_values
        mass_index = self.mass_index.output_values
        roc = self.roc.output_values
        rsi = self.rsi.output_values
        stc = self.stc.output_values
        # trix = self.trix.output_values
        tsi = self.tsi.output_values
        uo = self.uo.output_values

        aroon_up = [a.up if a is not None else None for a in self.aroon.output_values]
        aroon_down = [a.down if a is not None else None for a in self.aroon.output_values]
        adx_plus_di = [adx_val.plus_di if adx_val is not None else None for adx_val in self.adx.output_values]
        adx_minus_di = [adx_val.minus_di if adx_val is not None else None for adx_val in self.adx.output_values]
        macd_hist = [m.histogram if m is not None else None for m in self.macd]
        macd_signal = [m.signal if m is not None else None for m in self.macd]
        macd_val = [m.macd if m is not None else None for m in self.macd]
        stoch_k = [s.k if s is not None else None for s in self.stoch.output_values]
        stoch_d = [s.d if s is not None else None for s in self.stoch.output_values]
        stoch_rsi_k = [s.k if s is not None else None for s in self.stoch_rsi.output_values]
        stoch_rsi_d = [s.d if s is not None else None for s in self.stoch_rsi.output_values]
        # super_trend = [s.value if s is not None else None for s in self.super_trend.output_values]
        ttm_histogram = [t.histogram if t is not None else None for t in self.ttm.output_values]
        vtx_plus = [v.plus_vtx if v is not None else None for v in self.vtx.output_values]
        vtx_minus = [v.minus_vtx if v is not None else None for v in self.vtx.output_values]

        df['adx_plus_di'] = adx_plus_di
        df['adx_minus_di'] = adx_minus_di
        df['ao'] = ao
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        # df['atr'] = atr
        df['bop'] = bop
        df['chop'] = chop
        df['coppock_curve'] = coppock_curve
        df['dpo'] = dpo
        df['emv'] = emv
        df['ibs'] = ibs
        df['macd_hist'] = macd_hist
        df['macd_signal'] = macd_signal
        df['macd_val'] = macd_val
        df['mass_index'] = mass_index
        df['roc'] = roc
        df['rsi'] = rsi
        df['stc'] = stc
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_rsi_k'] = stoch_rsi_k
        df['stoch_rsi_d'] = stoch_rsi_d
        # df['super_trend'] = super_trend
        # df['trix'] = trix
        df['tsi'] = tsi
        df['ttm_histogram'] = ttm_histogram
        df['uo'] = uo
        df['vtx_plus'] = vtx_plus
        df['vtx_minus'] = vtx_minus
        
        df = self.create_shifted_features(df)
        df['symbol'] = self.symbol

        self.features_df = df

    def create_daily_features(self, daily_ohlcv: pd.DataFrame):
        if daily_ohlcv.shape[0] == 0:
            return
        self.org_df = pd.concat([self.org_df, daily_ohlcv]).reset_index(drop=True)

        new_df = self.org_df.iloc[-41:,:].reset_index(drop=True)
        new_df = self.create_basic_ohlcv_features(new_df)
        new_df = self.create_candle_related_features(new_df)
        new_df = new_df.iloc[-1,:]
        ohlcv = OHLCVFactory.from_dict({
            "open": daily_ohlcv['open'],
            "high": daily_ohlcv['high'],
            "low": daily_ohlcv['low'],
            "close": daily_ohlcv['close'],
            "volume": daily_ohlcv['volume']
        })
        close = daily_ohlcv['close'].tolist()

        self.adx.add(ohlcv)
        self.ao.add(ohlcv)
        self.aroon.add(ohlcv)
        # self.atr.add(ohlcv)
        self.bop.add(ohlcv)
        self.chop.add(ohlcv)
        self.coppock_curve.add(close)
        self.dpo.add(close)
        self.emv.add(ohlcv)
        self.ibs.add(ohlcv)
        self.macd.add(close)
        self.mass_index.add(ohlcv)
        self.roc.add(close)
        self.rsi.add(close)
        self.stc.add(close)
        self.stoch.add(ohlcv)
        self.stoch_rsi.add(close)
        # self.super_trend.add(ohlcv)
        # self.trix.add(close)
        self.tsi.add(close)
        self.ttm.add(ohlcv)
        self.uo.add(ohlcv)
        self.vtx.add(ohlcv)

        new_df['adx_plus_di'] = self.adx.output_values[-1].plus_di
        new_df['adx_minus_di'] = self.adx.output_values[-1].minus_di
        new_df['ao'] = self.ao.output_values[-1]
        new_df['aroon_up'] = self.aroon.output_values[-1].up
        new_df['aroon_down'] = self.aroon.output_values[-1].down
        # new_df['atr'] = self.atr.output_values[-1]
        new_df['bop'] = self.bop.output_values[-1]
        new_df['chop'] = self.chop.output_values[-1]
        new_df['coppock_curve'] = self.coppock_curve.output_values[-1]
        new_df['dpo'] = self.dpo.output_values[-1]
        new_df['emv'] = self.emv.output_values[-1]
        new_df['ibs'] = self.ibs.output_values[-1]
        new_df['macd_hist'] = self.macd.output_values[-1].histogram
        new_df['macd_signal'] = self.macd.output_values[-1].signal
        new_df['macd_val'] = self.macd.output_values[-1].macd
        new_df['mass_index'] = self.mass_index.output_values[-1]
        new_df['roc'] = self.roc.output_values[-1]
        new_df['rsi'] = self.rsi.output_values[-1]
        new_df['stc'] = self.stc.output_values[-1]
        new_df['stoch_k'] = self.stoch.output_values[-1].k
        new_df['stoch_d'] = self.stoch.output_values[-1].d
        new_df['stoch_rsi_k'] = self.stoch_rsi.output_values[-1].k
        new_df['stoch_rsi_d'] = self.stoch_rsi.output_values[-1].d
        # new_df['super_trend'] = self.super_trend.output_values[-1].value
        # new_df['trix'] = self.trix.output_values[-1]
        new_df['tsi'] = self.tsi.output_values[-1]
        new_df['ttm_histogram'] = self.ttm.output_values[-1].histogram
        new_df['uo'] = self.uo.output_values[-1]
        new_df['vtx_plus'] = self.vtx.output_values[-1].plus_vtx
        new_df['vtx_minus'] = self.vtx.output_values[-1].minus_vtx

        df = pd.concat([self.features_df, pd.DataFrame([new_df])]).reset_index(drop=True)
        df = self.create_shifted_features(df)
        df['symbol'] = self.symbol

        self.features_df = df

    def label(self):
        if self.features_df is None:
            raise ValueError("Features dataframe is not created yet.")

        upper_rate = 1.05
        self.features_df['next_close'] = self.features_df['close'].rolling(self.predicted_days).max().shift(-self.predicted_days)
        self.features_df['next_min_close'] = self.features_df['close'].rolling(self.predicted_days).min().shift(-self.predicted_days)

        self.features_df['label'] = self.features_df['next_close'] > self.features_df['open'].shift(-1) * upper_rate
        self.features_df['label'] &= self.features_df['next_min_close'] > self.features_df['open'].shift(-1) * 0.95
        
        self.features_df['weight'] = 1+(self.features_df['next_close']/self.features_df['open'].shift(-1)-1).abs()

def create_features(data, from_scratch: bool, features_data: dict[str, Features] = None):
    if not from_scratch and features_data is None:
        raise ValueError("features_data must be provided when from_scratch is False.")
    
    features_data = features_data or {}
    for symbol, df in tqdm(data.items()):
        if from_scratch:
            if USE_CACHE:
                cache_file = f'{CACHE_DIR}/{symbol}_features.pkl'
                if os.path.exists(cache_file):
                    features = pickle.load(open(cache_file, 'rb'))
            else:
                features = Features(symbol=symbol, org_df=df)
                features.create_features_from_scratch()
                features.label()
                pickle.dump(features, open(f'{CACHE_DIR}/{symbol}_features.pkl', 'wb'))
            features_data[symbol] = features
        else:
            features_data[symbol].create_daily_features(daily_ohlcv=df)
            features_data[symbol].label()
    
    return features_data


def remove_cols(df, return_removed_cols: bool = False):
    rm_cols = ['open', 'high', 'close', 'low', 'volume', 'time', 'symbol', 'next_close', 'next_min_close']
    rm_cols = [c for c in rm_cols if c in df.columns]
    if return_removed_cols:
        removed_cols_df = df[rm_cols]
    for c in rm_cols:
        if c in df.columns:
            del df[c]
    
    if return_removed_cols:
        return df, removed_cols_df
    return df


def get_current_features(features_data: dict[str, Features], current_time: datetime, volume_threshold: float = 5e6, return_removed_cols: bool = True):
    concat_features = [f.features_df[f.features_df['time'] == current_time] for f in features_data.values()]
    concat_features = pd.concat(concat_features)
    
    concat_features = concat_features[(abs(concat_features['close']/concat_features['open']-1) > 0.04) | (abs(concat_features['high']/concat_features['low']-1) > 0.08) | (abs(concat_features['cur_volume_vs_avg10_diff']) > 1.)]
    concat_features = concat_features[(concat_features['volume']*concat_features['close'] > volume_threshold)]
    
    concat_features.reset_index(drop=True, inplace=True)
    features_df, removed_cols_df = remove_cols(concat_features, return_removed_cols=return_removed_cols)
    if features_df.shape[0] != 0:
        if 'weight' in features_df.columns:
            del features_df['weight']
        del features_df['label']
    if features_df.isna().sum().sum() > 0:
        raise ValueError(f"Features dataframe contains NaN values: {[(c, features_df[c].isna().sum()) for c in features_df.columns if features_df[c].isna().sum() != 0]}")
    return features_df, removed_cols_df