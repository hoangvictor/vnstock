import pandas as pd
from scipy.stats import linregress

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from talipp.ohlcv import OHLCVFactory
from talipp.indicators import ADX, AO, Aroon, ATR, BOP, CHOP, CoppockCurve, DPO, EMV, IBS, \
    MACD, MassIndex, ROC, RSI, STC, Stoch, StochRSI, SuperTrend, TRIX, TSI, TTM, UO, VTX


class Features:
    def __init__(self, symbol: str, org_df: pd.DataFrame):
        self.symbol = symbol
        self.org_df = org_df
        self.features_df = None
    
    @staticmethod
    def estimate_increasing_rate(x):
        d = len(x)
        lr = linregress(range(d), x/x.tolist()[-1])
        i = lr.intercept.tolist()
        s = lr.slope.tolist()
        if i == 0:
            return s*d
        return s*d/i

    def create_features_from_scratch(self):

        df = self.org_df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        df['close_pct_change'] = df['close'].pct_change()
        df['close_open_diff'] = (df['close'] - df['open']) / df['open']
        df['high_low_diff'] = (df['high'] - df['low']) / df['low']
        df['close_low_diff'] = (df['close'] - df['low']) / df['low']
        df['volume_pct_change'] = df['volume'].pct_change()

        for col in ['close', 'volume']:
            for d in [5,10,20]:
                df[f'{col}_avg{d}'] = df[col].rolling(window=d).mean()
                df[f'cur_{col}_vs_avg{d}_diff'] = (df[col] - df[f'{col}_avg{d}']) / df[f'{col}_avg{d}']
                del df[f'{col}_avg{d}']

        for col in ['volume', 'close']:
            for d in [5, 10, 15]:
                df[f'{col}_slope{d}'] = df[col].rolling(d).apply(self.estimate_increasing_rate)

        for col in ['volume', 'close']:
            df[f'{col}_slope5_vs_slope10_diff'] = (1+df[f'{col}_slope5']) / (1+df[f'{col}_slope10'])-1
            df[f'{col}_slope5_vs_slope15_diff'] = (1+df[f'{col}_slope5']) / (1+df[f'{col}_slope15'])-1
            df[f'{col}_slope10_vs_slope15_diff'] = (1+df[f'{col}_slope10']) / (1+df[f'{col}_slope15'])-1

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
        self.atr = ATR(10, ohlcv)
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
        self.super_trend = SuperTrend(10, 3, ohlcv)
        self.trix = TRIX(18, close)
        self.tsi = TSI(13, 25, close)
        self.ttm = TTM(20, input_values = ohlcv)
        self.uo = UO(7, 10, 20, ohlcv)
        self.vtx = VTX(10, ohlcv)

        ao = self.ao.output_values
        atr = self.atr.output_values
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
        trix = self.trix.output_values
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
        super_trend = [s.value if s is not None else None for s in self.super_trend.output_values]
        ttm_histogram = [t.histogram if t is not None else None for t in self.ttm.output_values]
        vtx_plus = [v.plus_vtx if v is not None else None for v in self.vtx.output_values]
        vtx_minus = [v.minus_vtx if v is not None else None for v in self.vtx.output_values]

        df['adx_plus_di'] = adx_plus_di
        df['adx_minus_di'] = adx_minus_di
        df['ao'] = ao
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['atr'] = atr
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
        df['super_trend'] = super_trend
        df['trix'] = trix
        df['tsi'] = tsi
        df['ttm_histogram'] = ttm_histogram
        df['uo'] = uo
        df['vtx_plus'] = vtx_plus
        df['vtx_minus'] = vtx_minus
        
        shift_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        for shift_day in range(1,4):
            df[[f'prev{shift_day}d_{sc}' for sc in shift_cols]] = df.shift(shift_day)[shift_cols]

        df['symbol'] = self.symbol

        self.features_df = df

    def create_daily_features(self, daily_ohlcv: pd.DataFrame):
        if daily_ohlcv.shape[0] == 0:
            return
        self.org_df = pd.concat([self.org_df, daily_ohlcv]).reset_index(drop=True)

        new_df = self.org_df.iloc[-20:,:].reset_index(drop=True)
        new_df['time'] = pd.to_datetime(new_df['time'])
        new_df['close_pct_change'] = new_df['close'].pct_change()
        new_df['close_open_diff'] = (new_df['close'] - new_df['open']) / new_df['open']
        new_df['high_low_diff'] = (new_df['high'] - new_df['low']) / new_df['low']
        new_df['close_low_diff'] = (new_df['close'] - new_df['low']) / new_df['low']
        new_df['volume_pct_change'] = new_df['volume'].pct_change()

        for col in ['close', 'volume']:
            for d in [5,10,20]:
                new_df[f'{col}_avg{d}'] = new_df[col].rolling(window=d).mean()
                new_df[f'cur_{col}_vs_avg{d}_diff'] = (new_df[col] - new_df[f'{col}_avg{d}']) / new_df[f'{col}_avg{d}']
                del new_df[f'{col}_avg{d}']

        for col in ['volume', 'close']:
            for d in [5, 10, 15]:
                new_df[f'{col}_slope{d}'] = new_df[col].rolling(d).apply(self.estimate_increasing_rate)

        for col in ['volume', 'close']:
            new_df[f'{col}_slope5_vs_slope10_diff'] = (1+new_df[f'{col}_slope5']) / (1+new_df[f'{col}_slope10'])-1
            new_df[f'{col}_slope5_vs_slope15_diff'] = (1+new_df[f'{col}_slope5']) / (1+new_df[f'{col}_slope15'])-1
            new_df[f'{col}_slope10_vs_slope15_diff'] = (1+new_df[f'{col}_slope10']) / (1+new_df[f'{col}_slope15'])-1

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
        self.atr.add(ohlcv)
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
        self.super_trend.add(ohlcv)
        self.trix.add(close)
        self.tsi.add(close)
        self.ttm.add(ohlcv)
        self.uo.add(ohlcv)
        self.vtx.add(ohlcv)

        new_df['adx_plus_di'] = self.adx.output_values[-1].plus_di
        new_df['adx_minus_di'] = self.adx.output_values[-1].minus_di
        new_df['ao'] = self.ao.output_values[-1]
        new_df['aroon_up'] = self.aroon.output_values[-1].up
        new_df['aroon_down'] = self.aroon.output_values[-1].down
        new_df['atr'] = self.atr.output_values[-1]
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
        new_df['super_trend'] = self.super_trend.output_values[-1].value
        new_df['trix'] = self.trix.output_values[-1]
        new_df['tsi'] = self.tsi.output_values[-1]
        new_df['ttm_histogram'] = self.ttm.output_values[-1].histogram
        new_df['uo'] = self.uo.output_values[-1]
        new_df['vtx_plus'] = self.vtx.output_values[-1].plus_vtx
        new_df['vtx_minus'] = self.vtx.output_values[-1].minus_vtx

        df = pd.concat([self.features_df, pd.DataFrame([new_df])]).reset_index(drop=True)
        shift_cols = [c for c in self.features_df if c.startswith('prev')]
        for shift_day in range(1,4):
            df[[f'prev{shift_day}d_{sc}' for sc in shift_cols]] = df.shift(shift_day)[shift_cols]

        df['symbol'] = self.symbol

        self.features_df = df

    def label(self):
        if self.features_df is None:
            raise ValueError("Features dataframe is not created yet.")
    
        upper_rate = 1.05
        # df['label'] = df['close'].shift(-3) / df['close'] - 1
        self.features_df['next_close'] = self.features_df['close'].shift(-3)
        self.features_df['label'] = self.features_df['next_close'] > self.features_df['close'] * upper_rate
        # df['label2'] = df['close'].shift(-3) > df['close']


def create_features(data, from_scratch: bool, features_data: dict[str, Features] = None):
    if not from_scratch and features_data is None:
        raise ValueError("features_data must be provided when from_scratch is False.")
    
    features_data = features_data or {}
    for symbol, df in data.items():
        if from_scratch:
            features = Features(symbol=symbol, org_df=df)
            features.create_features_from_scratch()
            features.label()
            features_data[symbol] = features
        else:
            features_data[symbol].create_daily_features(daily_ohlcv=df)
            features_data[symbol].label()
    
    return features_data