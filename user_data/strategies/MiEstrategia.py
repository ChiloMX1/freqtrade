# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as ta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy import BooleanParameter, DecimalParameter
from freqtrade.strategy import informative
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.strategy import IntParameter

from freqtrade.strategy import (
    merge_informative_pair,
    stoploss_from_open,
    IntParameter,
    DecimalParameter,
    BooleanParameter,
)

def crossed_above(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 > series2)

def crossed_below(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 < series2)

class MiEstrategia(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False

    # ROI mínimo por trade: microganancias escalables
    minimal_roi = {
        "40": 0.003,   # 0.3%
        "20": 0.005,   # 0.5%
        "0": 0.065     # 6.5% fallback
    }

    # Stoploss ajustado a -1%
    stoploss = -0.01

    # Trailing stop activo para asegurar ganancias pequeñas
    trailing_stop = True
    trailing_stop_positive = 0.002     # Activa trailing a partir de 0.2%
    trailing_stop_positive_offset = 0.004  # Sigue ganando hasta 0.4%
    trailing_only_offset_is_reached = True

    # Comportamiento general
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    # RSI personalizado
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Tipos de orden ajustados para velocidad
    order_types = {
        "entry": "market",
        "exit": "market",     # ⚠ ahora usamos 'market' para evitar pérdidas por timeout
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
        if dataframe.empty:
            return dataframe

        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)
        dataframe["ema50"] = ta.ema(dataframe["close"], length=50)
        dataframe["tema"] = ta.tema(dataframe["close"], length=9)

        macd = ta.macd(dataframe["close"])
        if not macd.empty and macd.shape[1] >= 3:
            dataframe["macd"] = macd.iloc[:, 0]
            dataframe["macdsignal"] = macd.iloc[:, 1]
            dataframe["macdhist"] = macd.iloc[:, 2]

        dataframe["mfi"] = ta.mfi(
            high=dataframe["high"].astype(float),
            low=dataframe["low"].astype(float),
            close=dataframe["close"].astype(float),
            volume=dataframe["volume"].astype(float)
        )

        dataframe["volume_mean"] = dataframe["volume"].rolling(window=24).mean()

        bbands = ta.bbands(dataframe["close"], length=20, std=2)
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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (crossed_above(dataframe["rsi"], self.buy_rsi.value)) &
                (dataframe["tema"] <= dataframe["bb_middleband"]) &
                (dataframe["tema"] > dataframe["tema"].shift(1)) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (crossed_above(dataframe["rsi"], self.sell_rsi.value)) &
                (dataframe["tema"] > dataframe["bb_middleband"]) &
                (dataframe["tema"] < dataframe["tema"].shift(1)) &
                (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1
        return dataframe