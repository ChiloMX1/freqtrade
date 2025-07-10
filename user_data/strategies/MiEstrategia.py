# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import pandas_ta as ta


def crossed_above(series1, series2):
    if isinstance(series2, (int, float)):
        return (series1.shift(1) < series2) & (series1 > series2)
    else:
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)




class MiEstrategia(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short: bool = False

    minimal_roi = {
        "60": 0.035,
        "30": 0.050,
        "0": 0.075
    }

    stoploss = -0.075

    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.075
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    order_types = {
        "entry": "market",
        "exit": "limit",
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
            "main_plot": {
                "tema": {},

            },
            "subplots": {
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                }
            }
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Asegúrate de que el dataframe no esté vacío
        if dataframe.empty:
            return dataframe

        # RSI (Relative Strength Index)
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # EMA 50 (Exponential Moving Average)
        dataframe["ema50"] = ta.ema(dataframe["close"], length=50)

        # TEMA (Triple Exponential Moving Average)
        dataframe["tema"] = ta.tema(dataframe["close"], length=9)

        # MACD y sus componentes (devuelve 3 columnas)
        macd = ta.macd(dataframe["close"])
        if not macd.empty and macd.shape[1] >= 3:
            dataframe["macd"] = macd.iloc[:, 0]
            dataframe["macdsignal"] = macd.iloc[:, 1]
            dataframe["macdhist"] = macd.iloc[:, 2]

        # MFI (Money Flow Index) — asegúrate de que los tipos de datos sean float
        dataframe["mfi"] = ta.mfi(
            high=dataframe["high"].astype(float),
            low=dataframe["low"].astype(float),
            close=dataframe["close"].astype(float),
            volume=dataframe["volume"].astype(float)
        )

        # Media del volumen en 24 velas
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=24).mean()

        # Bandas de Bollinger
        bbands = ta.bbands(dataframe["close"], length=20, std=2)
        if not bbands.empty:
            dataframe["bb_lowerband"] = bbands["BBL_20_2.0"]
            dataframe["bb_middleband"] = bbands["BBM_20_2.0"]
            dataframe["bb_upperband"] = bbands["BBU_20_2.0"]

            # Porcentaje dentro de las bandas
            dataframe["bb_percent"] = (
                (dataframe["close"] - dataframe["bb_lowerband"]) /
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
            )

            # Ancho de las bandas
            dataframe["bb_width"] = (
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
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
