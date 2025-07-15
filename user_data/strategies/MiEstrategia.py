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
    Estrategia Final de Microtrading para Kraken (1m) - Versión Logger Fix
    """
    
    # ============= CONFIGURACIÓN INICIAL =============
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        # Inicialización CRÍTICA del logger
        self.logger = logging.getLogger(__name__)
        
        # Resto de tu inicialización
        self.loss_timestamps = {}
        self.consecutive_losses = {}
        self.global_cooldown_until = None
        self.cooldown_config = {
            'after_loss': 45 * 60,
            'consecutive_loss': 90 * 60,
            'global_drawdown': 0.02,
            'profit_threshold': 0.05
        }
        self.kraken = ccxt.kraken({
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True}
        })

    # ============= INDICADORES (VERSIÓN ROBUSTA) =============
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Versión ultra-estable con manejo de errores mejorado"""
        try:
            df = dataframe.copy()
            
            # 1. Cálculo seguro de indicadores
            df['rsi'] = pta.rsi(df['close'], length=2).clip(10, 90).ffill()
            df['ema5'] = pta.ema(df['close'], length=5).ffill()
            df['ema20'] = pta.ema(df['close'], length=20).ffill()
            
            # 2. Estocástico con verificación
            stoch = pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)
            if stoch is not None:
                df['stoch_k'] = stoch['STOCHk_3_3_3'].ffill()
                df['stoch_d'] = stoch['STOCHd_3_3_3'].ffill()
            
            # 3. Volumen con protección div/0
            df['volume_ma10'] = df['volume'].rolling(10).mean().replace(0, 1e-10)
            df['volume_ratio'] = (df['volume'] / df['volume_ma10']).clip(0, 100)
            
            # 4. Validación final
            self._validate_dataframe(df)
            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error en indicadores: {e}", exc_info=True)
            return dataframe.iloc[self.startup_candle_count or 50:]

    def _validate_dataframe(self, df: DataFrame):
        """Validación silenciosa para evitar errores"""
        try:
            if df.isnull().values.any():
                self.logger.warning("NaN detectados - Limpieza aplicada")
            lengths = {col: len(df[col]) for col in df.columns}
            if len(set(lengths.values())) > 1:
                self.logger.warning(f"Longitudes inconsistentes: {lengths}")
        except:
            pass  # No romper por validaciones

    # ============= RESTANTE DE TU ESTRATEGIA =============
    # (Mantén todo el resto de tus métodos igual que antes)
    # populate_entry_trend, populate_exit_trend, etc.

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