# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as pta
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from functools import reduce
from freqtrade.strategy import IntParameter
import logging

class MiEstrategia(IStrategy):
    """
    Estrategia corregida para Render + Kraken
    - Soluciona errores de condiciones pandas
    - Mantiene integridad de DataFrames
    - Optimizada para trading en vivo
    """

    # Configuración base (original)
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False
    startup_candle_count = 210

    minimal_roi = {
        "0": 0.065,
        "20": 0.005,
        "40": 0.003
    }

    stoploss = -0.01
    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.004
    trailing_only_offset_is_reached = True

    buy_rsi = IntParameter(10, 40, default=30, space='buy')
    sell_rsi = IntParameter(60, 90, default=70, space='sell')

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.trailing_active = False
        self.loss_timestamps = {}

    def _validate_df(self, df: DataFrame) -> bool:
        """Validación robusta del DataFrame"""
        return (isinstance(df, pd.DataFrame) and 
                not df.empty and 
                isinstance(df.index, pd.DatetimeIndex) and
                all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']))

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Cálculo seguro de indicadores"""
        try:
            if not self._validate_df(dataframe):
                return dataframe

            df = dataframe.copy()
            
            # Indicadores principales
            df['rsi'] = pta.rsi(df['close'], length=14).clip(0, 100)
            
            for length in [9, 21, 50, 200]:
                df[f'ema_{length}'] = pta.ema(df['close'], length=length)

            macd = pta.macd(df['close'])
            df[['macd', 'macdsignal', 'macdhist']] = macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error en indicators: {e}", exc_info=True)
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Señales de entrada corregidas"""
        try:
            if not self._validate_df(dataframe):
                return dataframe

            df = dataframe.copy()
            df['enter_long'] = 0

            # Construcción SEGURA de condiciones
            conditions = []
            
            if 'rsi' in df.columns:
                conditions.append(df['rsi'] > self.buy_rsi.value)
            
            if all(col in df.columns for col in ['close', 'ema_200']):
                conditions.append(df['close'] > df['ema_200'])
            
            if 'volume' in df.columns:
                conditions.append(df['volume'] > df['volume'].rolling(20).mean() * 0.7)

            # Aplicar solo si hay condiciones válidas
            if conditions:
                combined_cond = reduce(lambda x, y: x & y, conditions)
                df.loc[combined_cond, 'enter_long'] = 1

            return df[df.index.isin(dataframe.index)]

        except Exception as e:
            self.logger.error(f"Error en entry: {e}")
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Señales de salida corregidas"""
        try:
            if not self._validate_df(dataframe):
                return dataframe

            df = dataframe.copy()
            df['exit_long'] = 0

            exit_conditions = []
            
            if 'rsi' in df.columns:
                exit_conditions.append(df['rsi'] > self.sell_rsi.value)
            
            if all(col in df.columns for col in ['close', 'ema_9']):
                exit_conditions.append(df['close'] < df['ema_9'])
            
            if 'volume' in df.columns:
                exit_conditions.append(df['volume'] < df['volume'].rolling(20).mean() * 1.3)

            if exit_conditions:
                exit_cond = reduce(lambda x, y: x & y, exit_conditions)
                df.loc[exit_cond, 'exit_long'] = 1

            return df[df.index.isin(dataframe.index)]

        except Exception as e:
            self.logger.error(f"Error en exit: {e}")
            return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """Lógica de salida personalizada"""
        try:
            duration = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            if duration > 3 and current_profit < 0.002:
                return 'timeout_exit'
                
            if current_profit > 0.004:
                self.trailing_active = True
                return None
                
            if current_profit < -0.005:
                self.loss_timestamps[pair] = current_time
                return 'stop_loss'
                
            return None

        except Exception as e:
            self.logger.error(f"Error en custom_exit: {e}")
            return None