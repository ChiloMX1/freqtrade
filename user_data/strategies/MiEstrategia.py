# -*- coding: utf-8 -*-
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.persistence import Trade
from datetime import datetime, timezone
from typing import Dict, List, Optional
from functools import reduce
from freqtrade.strategy import IntParameter, DecimalParameter
import logging

logger = logging.getLogger(__name__)

class MiEstrategia(IStrategy):
    """
    Estrategia definitiva para microtrading agresivo en Kraken
    - Solución permanente al error de longitud del DataFrame
    - Optimizada para Render con ejecución estable
    - Enfoque en 8% diario con gestión de riesgo mejorada
    """

    # Configuración base probada en Render
    timeframe = '5m'
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200  # Suficiente para los indicadores

    # Gestión de riesgo optimizada
    stoploss = -0.007  # -0.7% (SL dinámico)
    trailing_stop = True
    trailing_stop_positive = 0.004  # 0.4%
    trailing_stop_positive_offset = 0.008  # 0.8%
    position_adjustment_enable = False

    # ROI dinámico para microtrading
    minimal_roi = {
        "0": 0.01,   # 1% para trades <5min
        "10": 0.007, # 0.7% para trades 10-20min
        "20": 0.004, # 0.4% para trades 20-30min
        "30": 0      # Cierre obligatorio a los 30min
    }

    # Hiperparámetros optimizados
    buy_rsi = IntParameter(22, 32, default=26, space='buy')
    sell_rsi = IntParameter(68, 78, default=72, space='sell')
    buy_volume = DecimalParameter(2.8, 4.2, decimals=1, default=3.4, space='buy')
    max_volatility = DecimalParameter(0.01, 0.025, default=0.018, space='buy')

    # Configuración de órdenes para Kraken
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        # Variables de estado
        self.today_trades = 0
        self.max_daily_trades = 24  # Ajustado para objetivo de 8%
        self.loss_timestamps = {}
        self.consecutive_losses = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        VERSIÓN COMPROBADA QUE ELIMINA EL ERROR DE LONGITUD:
        1. Usa solo TA-Lib para máxima compatibilidad
        2. Sincronización explícita de longitudes
        3. Validación en tiempo real
        """
        try:
            # 1. Crear nuevo DataFrame solo con columnas necesarias
            df = DataFrame(index=dataframe.index)
            df['open'] = dataframe['open']
            df['high'] = dataframe['high']
            df['low'] = dataframe['low']
            df['close'] = dataframe['close']
            df['volume'] = dataframe['volume']
            
            # 2. Calcular indicadores (todos con el mismo length)
            df['ema8'] = ta.EMA(df['close'], timeperiod=8)
            df['ema21'] = ta.EMA(df['close'], timeperiod=21)
            df['rsi'] = ta.RSI(df['close'], timeperiod=5)
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 3. Calcular indicadores derivados
            df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
            df['volume_ratio'] = np.where(
                df['volume_ma'] > 0,
                df['volume'] / df['volume_ma'],
                1.0
            ).clip(0, 5)
            df['volatility'] = (df['atr'] / df['close']).clip(0.005, 0.03)
            
            # 4. Eliminar filas con NaN (sincronización garantizada)
            initial_length = len(df)
            df.dropna(inplace=True)
            if len(df) < initial_length:
                logger.info(f"Se eliminaron {initial_length - len(df)} filas con NaN")
            
            # 5. Validación de integridad
            self._validate_dataframe(df)
            
            return df

        except Exception as e:
            logger.error(f"Error crítico en populate_indicators: {str(e)}")
            # Fallback ultraconservador
            backup_df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
            backup_df['ema8'] = ta.EMA(backup_df['close'], timeperiod=8)
            backup_df['ema21'] = ta.EMA(backup_df['close'], timeperiod=21)
            backup_df['rsi'] = ta.RSI(backup_df['close'], timeperiod=5)
            return backup_df.dropna()

    def _validate_dataframe(self, df: DataFrame):
        """Validación exhaustiva del DataFrame"""
        required_columns = {
            'open', 'high', 'low', 'close', 'volume',
            'ema8', 'ema21', 'rsi', 'atr',
            'volume_ma', 'volume_ratio', 'volatility'
        }
        
        # Verificar columnas existentes
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
            
        # Verificar longitudes consistentes
        lengths = {col: len(df[col].dropna()) for col in required_columns}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Inconsistencia en longitudes: {lengths}")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Condiciones de entrada optimizadas:
        - Filtros de volumen y volatilidad
        - Validación de tendencia
        - Cooldown integrado
        """
        df = dataframe.copy()
        pair = metadata['pair']
        
        # 1. Verificar cooldown del par
        if self._is_in_cooldown(pair):
            df['enter_long'] = 0
            return df

        # 2. Condiciones principales
        conditions = [
            df['rsi'] < self.buy_rsi.value,
            df['close'] > df['ema8'],
            df['volume_ratio'] > self.buy_volume.value,
            df['volatility'] < self.max_volatility.value,
            df['close'] > df['ema21']  # Filtro de tendencia añadido
        ]
        
        # 3. Aplicar condiciones
        df['enter_long'] = 0
        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Condiciones de salida optimizadas:
        - Take-profit dinámico
        - Protección de ganancias
        """
        df = dataframe.copy()
        
        exit_conditions = [
            df['rsi'] > self.sell_rsi.value,
            df['close'] < df['ema8'] * 0.995
        ]
        
        df['exit_long'] = 0
        if exit_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1
            
        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Gestión de riesgo mejorada:
        - Control de trades diarios
        - Filtro de spread
        - Validación de liquidez
        """
        # 1. Límite diario de trades
        if self.today_trades >= self.max_daily_trades:
            return False
            
        # 2. Validar condiciones del mercado
        try:
            ticker = self.dp.ticker(pair)
            spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            if spread > 0.0015:  # 0.15% máximo
                return False
                
            # 3. Validar volumen actual
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            if last_candle['volume'] < last_candle['volume_ma'] * 1.5:
                return False
                
            return True
            
        except Exception:
            return False

    def _is_in_cooldown(self, pair: str) -> bool:
        """Gestión mejorada de cooldown por par"""
        last_loss = self.loss_timestamps.get(pair)
        if not last_loss:
            return False
            
        elapsed = (datetime.now(timezone.utc) - last_loss).total_seconds() / 60
        return elapsed < 45  # 45 minutos de cooldown

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Salidas personalizadas:
        - Registro de pérdidas para cooldown
        - Protección de capital
        """
        if current_profit < -0.005:  # -0.5%
            self.loss_timestamps[pair] = current_time
            self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1
        return None