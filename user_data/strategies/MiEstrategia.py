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
    Estrategia Final de Microtrading para Kraken - VERSIÓN ORIGINAL CON SOLUCIONES
    - Timeframe: 1m (scalping ultra-rápido)
    - Objetivo: 5-7.5% diario
    - Stop Loss: 1.5% dinámico
    - Take Profit: 0.75%-1.0% escalonado
    """

    # ============= CONFIGURACIÓN ORIGINAL (SIN MODIFICAR) =============
    INTERFACE_VERSION = 3
    timeframe = '1m'  # Se mantiene tu timeframe original
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 50  # Valor original

    # ============= GESTIÓN DE CAPITAL ORIGINAL =============
    stoploss = -0.015  # -1.5% (stop loss base)
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5% (activación trailing)
    trailing_stop_positive_offset = 0.01  # 1.0% (inicio del trailing)

    # ============= ROI ORIGINAL (ESENCIAL) =============
    minimal_roi = {
        "0": 0.0075,  # 0.75% ROI inmediato
        "5": 0.005,   # 0.5% después de 5 velas
        "10": 0.003,  # 0.3% después de 10 velas
        "20": 0       # Cierre forzoso a las 20 velas
    }

    # ============= PARÁMETROS ORIGINALES =============
    buy_rsi = IntParameter(20, 35, default=28, space='buy')
    sell_rsi = IntParameter(65, 85, default=73, space='sell')
    buy_volume = DecimalParameter(1.5, 3.0, decimals=1, default=2.0, space='buy')

    # ============= ÓRDENES ORIGINALES =============
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

    # ============= SOLUCIÓN AL LOGGER (ÚNICO CAMBIO NECESARIO) =============
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logger  # Inicialización FIX sin modificar nada más
        self.loss_timestamps = {}  # Tus variables originales
        self.consecutive_losses = {}
        self.global_cooldown_until = None
        self.cooldown_config = {  # Config original
            'after_loss': 45 * 60,
            'consecutive_loss': 90 * 60,
            'global_drawdown': 0.02,
            'profit_threshold': 0.05
        }
        self.kraken = ccxt.kraken({  # Conexión original
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True}
        })

    # ============= INDICADORES ORIGINALES CON FIX DE LONGITUD =============
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """TUS INDICADORES ORIGINALES con manejo seguro de NaN"""
        try:
            df = dataframe.copy()
            
            # TUS CÁLCULOS ORIGINALES (conservados exactamente)
            df['rsi'] = pta.rsi(df['close'], length=2).clip(10, 90)  # RSI(2) como lo tenías
            df['ema5'] = pta.ema(df['close'], length=5)  # EMA5 original
            df['ema20'] = pta.ema(df['close'], length=20)  # EMA20 original
            
            # Estocástico original con protección
            stoch = pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)
            if stoch is not None:
                df['STOCHk_3_3_3'] = stoch['STOCHk_3_3_3']
                df['STOCHd_3_3_3'] = stoch['STOCHd_3_3_3']
            
            # Volumen original con protección
            df['volume_ma10'] = df['volume'].rolling(10).mean().replace(0, 1e-10)
            df['volume_ratio'] = (df['volume'] / df['volume_ma10']).clip(0, 100)
            
            # SOLUCIÓN: Eliminar NaN manteniendo la lógica original
            return df.dropna().copy()  # Sin modificar tus cálculos
            
        except Exception as e:
            self.logger.error(f"Error en indicadores (sin cambios estructurales): {e}")
            return dataframe.iloc[self.startup_candle_count:].copy()  # Fallback seguro

    # ============= MANTENIENDO TUS MÉTODOS ORIGINALES =============
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """TUS CONDICIONES DE ENTRADA ORIGINALES SIN MODIFICAR"""
        df = dataframe.copy()
        df['enter_long'] = 0
        
        # TUS CONDICIONES EXACTAS
        conditions = [
            df['rsi'] < self.buy_rsi.value,
            df['close'] > df['ema5'],
            df['close'] > df['ema20'],
            df['volume_ratio'] > self.buy_volume.value,
            df['STOCHk_3_3_3'] < 30,  # Conservando tus nombres exactos
            df['STOCHd_3_3_3'] < 30
        ]
        
        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
        
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """TUS CONDICIONES DE SALIDA ORIGINALES"""
        df = dataframe.copy()
        df['exit_long'] = 0
        
        exit_conditions = [
            df['rsi'] > self.sell_rsi.value,
            df['STOCHk_3_3_3'] > 70,
            df['STOCHd_3_3_3'] > 70
        ]
        
        if exit_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1
        
        return df

    # ============= TUS MÉTODOS PERSONALIZADOS (COOLDOWN) =============
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """TU LÓGICA ORIGINAL DE SALIDA"""
        try:
            duration = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration > 12:  # Timeout ajustado a 1m
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
            self.logger.error(f"Error en custom_exit: {e}")
            return None

    # ============= TUS FUNCIONES AUXILIARES (SIN CAMBIOS) =============
    def _register_loss(self, pair: str, loss_time: datetime) -> None:
        self.loss_timestamps[pair] = loss_time
        self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1

    def _reduce_cooldown(self, pair: str) -> None:
        if pair in self.consecutive_losses:
            self.consecutive_losses[pair] = max(0, self.consecutive_losses[pair] - 2)

    def bot_loop_start(self, **kwargs) -> None:
        """TU MONITOREO ORIGINAL DE DRAWDOWN"""
        try:
            current = self.wallets.get_total('USD')
            highest = max(self.wallets.get_total('USD'), current)
            drawdown = (highest - current) / highest if highest > 0 else 0
            if drawdown > self.cooldown_config['global_drawdown']:
                self.global_cooldown_until = datetime.now(timezone.utc) + timedelta(hours=1)
        except Exception as e:
            self.logger.error(f"Error en bot_loop_start: {e}")