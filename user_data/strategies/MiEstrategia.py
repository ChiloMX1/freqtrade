# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as ta

class MiEstrategia(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False

    # ROI y Stoploss
    minimal_roi = {
        "40": 0.04,
        "20": 0.05,
        "0": 0.065
    }

    stoploss = -0.075

    # Trailing Stop
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.055
    trailing_only_offset_is_reached = True

    # Config
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 50

    # Ordenes
    order_types = {
        "entry": "market",
        "exit": "limit",
        "stoploss": "market"
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        # Indicadores agresivos
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        dataframe['adx'] = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'])['ADX_14']
        dataframe['ema20'] = ta.ema(dataframe['close'], length=20)
        dataframe['ema50'] = ta.ema(dataframe['close'], length=50)
        dataframe['cci'] = ta.cci(dataframe['high'], dataframe['low'], dataframe['close'], length=20)
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 55) &
                (dataframe['adx'] > 25) &
                (dataframe['cci'] > 100) &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['ema20'] > dataframe['ema50'])
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 75) |
                (dataframe['cci'] < 0) |
                (dataframe['close'] < dataframe['ema20'])
            ),
            'exit_long'
        ] = 1
        return dataframe