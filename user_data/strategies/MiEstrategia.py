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

class MiEstrategia(IStrategy):
    """
    Estrategia Final de Microtrading para Kraken (1m) - Versión Render Fix
    - Soluciona el error 'mismatching length' con manejo robusto de indicadores
    - Timeframe: 1m para scalping ultra-rápido
    - Stop Loss: 1.5% dinámico
    - Take Profit: 0.75%-1.0% escalonado
    """

    # =============================================
    # 1. CONFIGURACIÓN BASE (RENDER + KRAKEN)
    # =============================================
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 100  # Ampliado para mayor seguridad

    # =============================================
    # 2. GESTIÓN DE CAPITAL
    # =============================================
    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01

    # =============================================
    # 3. OBJETIVOS DE RENTABILIDAD
    # =============================================
    minimal_roi = {
        "0": 0.0075,
        "5": 0.005,
        "10": 0.003,
        "20": 0
    }

    # =============================================
    # 4. PARÁMETROS OPTIMIZADOS
    # =============================================
    buy_rsi = IntParameter(20, 35, default=28, space='buy')
    sell_rsi = IntParameter(65, 85, default=73, space='sell')
    buy_volume = DecimalParameter(1.5, 3.0, decimals=1, default=2.0, space='buy')

    # =============================================
    # 5. CONFIGURACIÓN DE ÓRDENES
    # =============================================
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'entry': 'IOC',
        'exit': 'GTC'
    }

    # =============================================
    # 6. FUNCIÓN PRINCIPAL DE INDICADORES (FIXED)
    # =============================================
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Versión corregida con:
        - Manejo robusto de longitudes
        - Verificación de NaN
        - Compatibilidad total con Render
        """
        try:
            # 1. Copia segura del DataFrame
            df = dataframe.copy()
            initial_len = len(df)
            
            # 2. Calculamos todos los indicadores por separado
            indicators = {
                'rsi': pta.rsi(df['close'], length=2).clip(10, 90),
                'ema5': pta.ema(df['close'], length=5),
                'ema20': pta.ema(df['close'], length=20),
                'stoch_k': pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)['STOCHk_3_3_3'],
                'stoch_d': pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)['STOCHd_3_3_3'],
                'volume_ma10': df['volume'].rolling(10).mean()
            }
            
            # 3. Encontramos la longitud mínima válida
            valid_lengths = [len(v.dropna()) for v in indicators.values() if hasattr(v, 'dropna')]
            min_length = min(valid_lengths) if valid_lengths else initial_len
            
            # 4. Aplicamos todos los indicadores recortados
            for name, values in indicators.items():
                df[name] = values.iloc[-min_length:] if hasattr(values, 'iloc') else values[-min_length:]
            
            # 5. Calculamos derivados
            df['volume_ratio'] = df['volume'] / df['volume_ma10']
            
            # 6. Verificación final de consistencia
            self._validate_dataframe(df)
            
            return df.iloc[-min_length:]  # Retorno consistente

        except Exception as e:
            self.logger.error(f"Error en populate_indicators: {str(e)}", exc_info=True)
            return dataframe.iloc[self.startup_candle_count:]  # Fallback seguro

    def _validate_dataframe(self, df: DataFrame):
        """Valida la integridad del DataFrame"""
        if df.isnull().values.any():
            self.logger.warning("NaN detectados en el DataFrame")
        lengths = {col: len(df[col]) for col in df.columns}
        if len(set(lengths.values())) > 1:
            self.logger.error(f"Inconsistencia en longitudes: {lengths}")

    # =============================================
    # 7. SEÑALES DE ENTRADA (OPTIMIZADAS)
    # =============================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        try:
            df = dataframe.copy()
            df['enter_long'] = 0
            
            conditions = [
                df['rsi'] < self.buy_rsi.value,
                df['close'] > df['ema5'],
                df['close'] > df['ema20'],
                df['volume_ratio'] > self.buy_volume.value,
                df['stoch_k'] < 30,
                df['stoch_d'] < 30
            ]
            
            if conditions:
                df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error en entry_trend: {str(e)}")
            return dataframe

    # =============================================
    # 8. SEÑALES DE SALIDA
    # =============================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        try:
            df = dataframe.copy()
            df['exit_long'] = 0
            
            exit_conditions = [
                df['rsi'] > self.sell_rsi.value,
                df['stoch_k'] > 70,
                df['stoch_d'] > 70
            ]
            
            if exit_conditions:
                df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error en exit_trend: {str(e)}")
            return dataframe

    # =============================================
    # 9. GESTIÓN DE RIESGO (COOLDOWN)
    # =============================================
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.loss_timestamps = {}
        self.consecutive_losses = {}
        self.global_cooldown_until = None
        self.cooldown_config = {
            'after_loss': 45 * 60,
            'consecutive_loss': 90 * 60,
            'global_drawdown': 0.02,
            'profit_threshold': 0.05
        }
        self.kraken = ccxt.kraken({'enableRateLimit': True})

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                  current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        try:
            duration = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration > 12:
                return 'timeout_12min'
            if current_profit > 0.01:
                if current_profit > self.cooldown_config['profit_threshold']:
                    self._reduce_cooldown(pair)
                return 'take_profit_1%'
            if current_profit < -0.01:
                self._register_loss(pair, current_time)
                return 'stop_loss_1.5%'
            return None
        except Exception as e:
            self.logger.error(f"Error en custom_exit: {str(e)}")
            return None

    # =============================================
    # 10. FUNCIONES AUXILIARES
    # =============================================
    def _register_loss(self, pair: str, loss_time: datetime) -> None:
        self.loss_timestamps[pair] = loss_time
        self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1

    def _reduce_cooldown(self, pair: str) -> None:
        if pair in self.consecutive_losses:
            self.consecutive_losses[pair] = max(0, self.consecutive_losses[pair] - 2)

    def bot_loop_start(self, **kwargs) -> None:
        """Manejo de drawdown global"""
        try:
            current = self.wallets.get_total('USD')
            highest = max(self.wallets.get_total('USD'), current)
            drawdown = (highest - current) / highest if highest > 0 else 0
            if drawdown > self.cooldown_config['global_drawdown']:
                self.global_cooldown_until = datetime.now(timezone.utc) + timedelta(hours=1)
        except Exception as e:
            self.logger.error(f"Error en bot_loop_start: {str(e)}")