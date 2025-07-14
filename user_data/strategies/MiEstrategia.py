# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as pta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import reduce
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy import BooleanParameter, DecimalParameter
from freqtrade.strategy import informative
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.strategy import IntParameter


def crossed_above(series1, series2):
    if not hasattr(series1, "shift"):
        return False
    if isinstance(series2, (int, float)):
        return (series1.shift(1) < series2) & (series1 > series2)
    else:
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)


class MiEstrategia(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.trailing_active = False
        self.trailing_roi = 0.004
        self.loss_timestamps = {}
        self.cooldowns = {}
        self._last_valid_index = None

    def _validate_data_structure(self, dataframe: DataFrame) -> bool:
        """Validación exhaustiva del DataFrame"""
        if dataframe.empty:
            self.logger.debug("DataFrame vacío recibido")
            return False
        
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(dataframe.columns):
            missing = required_columns - set(dataframe.columns)
            self.logger.warning(f"Columnas OHLCV faltantes: {missing}")
            return False
        
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            self.logger.warning("Índice no es DatetimeIndex")
            return False
            
        if not dataframe.index.is_monotonic_increasing:
            self.logger.warning("Índice no es monotónico creciente")
            return False
        
        return True

INTERFACE_VERSION = 3
timeframe = "5m"
can_short = False

minimal_roi = {
    "40": 0.003,
    "20": 0.005,
    "0": 0.065
}

use_exit_signal = True
exit_profit_only = True
ignore_roi_if_entry_signal = False
stoploss = -0.01
trailing_stop = True
trailing_stop_positive = 0.002
trailing_stop_positive_offset = 0.004
trailing_only_offset_is_reached = True
process_only_new_candles = True
use_custom_exit = True
startup_candle_count: int = 210

buy_rsi = IntParameter(10, 40, default=30, space="buy")
sell_rsi = IntParameter(60, 90, default=70, space="sell")

order_types = {
    "entry": "market",
    "exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False
}

order_time_in_force = {
    "entry": "GTC",
    "exit": "GTC"
}

@property
def plot_config(self):
    return {
        "main_plot": {"tema": {}},
        "subplots": {
            "MACD": {"macd": {"color": "blue"}, "macdsignal": {"color": "orange"}},
            "RSI": {"rsi": {"color": "red"}}
        }
    }

def informative_pairs(self):
    return []

def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    try:
        if not self._validate_data_structure(dataframe):
            return dataframe
        
        dataframe = dataframe.copy()
        
        try:
            dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)
        except Exception as e:
            self.logger.warning(f"Error calculando RSI: {str(e)}")
            dataframe["rsi"] = np.nan
        
        try:
            ema_lengths = [9, 21, 50, 200]
            for length in ema_lengths:
                dataframe[f'ema_{length}'] = pta.ema(dataframe['close'], length=length)
        except Exception as e:
            self.logger.warning(f"Error calculando EMAs: {str(e)}")
        
        try:
            macd = pta.macd(dataframe["close"])
            if isinstance(macd, pd.DataFrame) and not macd.empty:
                dataframe["macd"] = macd["MACD_12_26_9"]
                dataframe["macdsignal"] = macd["MACDs_12_26_9"]
                dataframe["macdhist"] = macd["MACDh_12_26_9"]
        except Exception as e:
            self.logger.warning(f"Error calculando MACD: {str(e)}")
        
        try:
            dataframe["mfi"] = pta.mfi(
                high=dataframe["high"].astype(float),
                low=dataframe["low"].astype(float),
                close=dataframe["close"].astype(float),
                volume=dataframe["volume"].astype(float)
            )
        except Exception as e:
            self.logger.warning(f"Error calculando MFI: {str(e)}")
        
        try:
            dataframe["volume_mean"] = dataframe["volume"].rolling(window=24).mean()
        except Exception as e:
            self.logger.warning(f"Error calculando volumen medio: {str(e)}")
        
        try:
            bbands = pta.bbands(dataframe["close"], length=20, std=2)
            if not bbands.empty:
                dataframe["bb_lowerband"] = bbands["BBL_20_2.0"]
                dataframe["bb_middleband"] = bbands["BBM_20_2.0"]
                dataframe["bb_upperband"] = bbands["BBU_20_2.0"]
                dataframe["bb_percent"] = (
                    (dataframe["close"] - dataframe["bb_lowerband"]) /
                    (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
                )
                dataframe["bb_width"] = (
                    (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) /
                    dataframe["bb_middleband"]
                )
        except Exception as e:
            self.logger.warning(f"Error calculando Bollinger Bands: {str(e)}")
        
        if dataframe.isnull().values.any():
            dataframe = dataframe.dropna()
            
        return dataframe
    
    except Exception as e:
        self.logger.error(f"Error crítico en populate_indicators: {str(e)}")
        return DataFrame()

def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    try:
        if not self._validate_data_structure(dataframe):
            return dataframe
        
        dataframe = dataframe.copy()
        
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
            
        required_indicators = ['rsi', 'ema_9', 'ema_21', 'ema_200', 'volume']
        if not all(ind in dataframe.columns for ind in required_indicators):
            self.logger.warning(f"Indicadores faltantes para {metadata['pair']}")
            return dataframe
            
        conditions = []
        try:
            conditions.append(dataframe['rsi'] > 50)
            conditions.append(dataframe['volume'] > 0)
            conditions.append(dataframe['ema_9'] > dataframe['ema_21'])
            
            volatility = (dataframe['high'] - dataframe['low']).rolling(window=5).mean()
            conditions.append((dataframe['high'] - dataframe['low']) < volatility)
            
            conditions.append(dataframe['close'] > dataframe['ema_200'])
            
            if conditions:
                condition_mask = reduce(lambda x, y: x & y, conditions)
                dataframe.loc[condition_mask, 'enter_long'] = 1
                
        except Exception as e:
            self.logger.warning(f"Error aplicando condiciones de entrada: {str(e)}")
            
        return dataframe
        
    except Exception as e:
        self.logger.error(f"Error crítico en populate_entry_trend: {str(e)}")
        return dataframe

def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    try:
        if not self._validate_data_structure(dataframe):
            return dataframe
            
        if 'exit_long' not in dataframe.columns:
            dataframe['exit_long'] = 0

        try:
            exit_mask = (
                (crossed_above(dataframe["rsi"], self.sell_rsi.value)) &
                (dataframe["tema"] > dataframe["bb_middleband"]) &
                (dataframe["tema"] < dataframe["tema"].shift(1)) &
                (dataframe["volume"] > 0)
            )

            if not exit_mask.empty and exit_mask.any():
                dataframe.loc[exit_mask, "exit_long"] = 1

        except Exception as e:
            self.logger.warning(f"Error en condiciones de salida: {str(e)}")

        return dataframe
        
    except Exception as e:
        self.logger.error(f"Error crítico en populate_exit_trend: {str(e)}")
        return dataframe

def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs) -> Optional[str]:
    try:
        if not isinstance(current_time, datetime):
            self.logger.error("current_time no es datetime")
            return None
            
        if not isinstance(trade, Trade):
            self.logger.error("trade no es instancia de Trade")
            return None
            
        try:
            current_time = current_time.replace(tzinfo=timezone.utc)
            open_date = trade.open_date_utc.replace(tzinfo=timezone.utc)
            duration = (current_time - open_date).total_seconds() / 60
        except Exception as e:
            self.logger.error(f"Error calculando duración: {str(e)}")
            return None
            
        if duration > 180 and current_profit < 0.002:
            return "timeout_exit"
            
        if current_profit > 0.004:
            self.trailing_active = True
            self.trailing_roi = 0.0025
            
        if current_profit < 0:
            self.loss_timestamps[pair] = current_time
            
        return None
        
    except Exception as e:
        self.logger.error(f"Error inesperado en custom_exit: {str(e)}")
        return None

def cooldown_active(self, pair: str, current_time: datetime) -> bool:
    try:
        last_loss_time = self.loss_timestamps.get(pair)
        if last_loss_time:
            cooldown_duration = timedelta(minutes=45)
            if current_time - last_loss_time < cooldown_duration:
                return True
        return False
    except Exception as e:
        self.logger.error(f"Error en cooldown_active: {str(e)}")
        return False