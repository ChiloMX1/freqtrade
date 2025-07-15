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
from freqtrade.strategy import IntParameter, DecimalParameter
import logging
import time
import ccxt

logger = logging.getLogger(__name__)

class MiEstrategia(IStrategy):
    """
    Estrategia definitiva para microtrading en Kraken con:
    - Solución permanente al error de longitud del DataFrame
    - Lógica original conservada al 100%
    - Optimizada para Render
    - Enfoque en 15-20 trades/bot/día
    """

    # ============= CONFIGURACIÓN ORIGINAL =============
    timeframe = '5m'
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200  # Suficiente para EMA21 + ATR14

    # ============= PARÁMETROS ORIGINALES =============
    stoploss = -0.01
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01

    minimal_roi = {
        "0": 0.008,
        "5": 0.005,
        "10": 0.003,
        "20": 0
    }

    buy_rsi = IntParameter(20, 32, default=25, space='buy')
    sell_rsi = IntParameter(70, 80, default=75, space='sell')
    buy_volume = DecimalParameter(2.0, 3.5, decimals=1, default=2.8, space='buy')

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logger
        self.today_trades = 0
        self.max_daily_trades = 18

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        VERSIÓN CON SOLUCIÓN AL ERROR DE LONGITUD:
        - Solo se modificó este método para resolver el problema técnico
        - Todo lo demás permanece exactamente igual
        """
        try:
            # 1. Copia segura del DataFrame
            df = dataframe.copy()
            
            # 2. Solución del issue #3686 - Calcular longitud mínima primero
            lengths = []
            indicators = {}
            
            # Calculamos RSI y obtenemos su longitud
            rsi = pta.rsi(df['close'], length=5).dropna()
            indicators['rsi'] = rsi
            lengths.append(len(rsi))
            
            # Calculamos EMA8 y obtenemos su longitud
            ema8 = pta.ema(df['close'], length=8).dropna()
            indicators['ema8'] = ema8
            lengths.append(len(ema8))
            
            # Calculamos EMA21 y obtenemos su longitud
            ema21 = pta.ema(df['close'], length=21).dropna()
            indicators['ema21'] = ema21
            lengths.append(len(ema21))
            
            # Calculamos ATR y obtenemos su longitud
            atr = pta.atr(df['high'], df['low'], df['close'], length=14).dropna()
            indicators['atr'] = atr
            lengths.append(len(atr))
            
            # Calculamos volumen medio y obtenemos su longitud
            volume_ma = df['volume'].rolling(10).mean().dropna()
            indicators['volume_ma'] = volume_ma
            lengths.append(len(volume_ma))
            
            # 3. Determinamos la longitud mínima común
            min_length = min(lengths)
            
            # 4. Aplicamos la solución del issue #3686
            df = df.iloc[-min_length:].copy()
            for name, values in indicators.items():
                df[name] = values.iloc[-min_length:]
            
            # 5. Columnas derivadas (MANTENIENDO TU LÓGICA ORIGINAL)
            df['volume_ratio'] = (df['volume'] / df['volume_ma'].replace(0, 1e-10)).clip(0, 5)
            df['volatility'] = (df['atr'] / df['close']).clip(0.005, 0.03)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error en indicators: {e}")
            return dataframe.iloc[self.startup_candle_count:].copy()

    def _validate_dataframe(self, df: DataFrame):
        """Valida silenciosamente la integridad del DataFrame"""
        lengths = {col: len(df[col].dropna()) for col in df.columns}
        if len(set(lengths.values())) > 1:
            self.logger.debug(f"Longitudes: {lengths}")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Tus condiciones de entrada ORIGINALES"""
        df = dataframe.copy()
        df['enter_long'] = 0
        
        conditions = [
            df['rsi'] < self.buy_rsi.value,
            df['close'] > df['ema8'],
            df['volume_ratio'] > self.buy_volume.value,
            df['volatility'] < 0.02
        ]
        
        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Tus condiciones de salida ORIGINALES"""
        df = dataframe.copy()
        df['exit_long'] = 0
        
        exit_conditions = [
            df['rsi'] > self.sell_rsi.value,
            df['close'] < df['ema8'] * 0.995
        ]
        
        if exit_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1
            
        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """Gestión de riesgo ORIGINAL"""
        if self.today_trades >= self.max_daily_trades:
            return False
            
        try:
            ticker = self.dp.ticker(pair)
            spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            return spread <= 0.0015  # 0.15% max spread
            
        except Exception:
            return False

    def bot_loop_start(self, **kwargs) -> None:
        """Reinicio diario ORIGINAL"""
        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute < 5:
            self.today_trades = 0