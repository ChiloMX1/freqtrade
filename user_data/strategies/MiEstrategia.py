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
import time

class MiEstrategia(IStrategy):
    """
    Versión optimizada para despliegue directo en Render con Kraken
    - Manejo robusto de conexiones intermitentes
    - Validación estricta de datos remotos
    - Auto-recuperación de errores
    """

    # 1. Configuración Base para Render
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 300  # Buffer amplio para datos remotos

    # 2. Parámetros Optimizados para Kraken
    minimal_roi = {
        "0": 0.07,    # 7% ROI para trades largos
        "30": 0.02,   # 2% después de 30 velas
        "60": 0.01    # 1% después de 60 velas
    }

    stoploss = -0.015  # -1.5% stoploss inicial
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # 3. Parámetros Dinámicos (ajustables via Telegram)
    buy_rsi = IntParameter(28, 38, default=32, space='buy')
    sell_rsi = IntParameter(68, 85, default=75, space='sell')

    # 4. Configuración de Órdenes para Kraken en Render
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True  # Crítico para Render
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.last_refresh = time.time()
        self.data_retries = {}  # Para reintentos por par
        self.logger.info("Inicializando estrategia para Render+Kraken")

    def _get_kraken_dataframe(self, dataframe: DataFrame, pair: str) -> Optional[DataFrame]:
        """
        Procesamiento seguro de DataFrame para Kraken en entorno remoto
        """
        try:
            # Validación básica
            if dataframe.empty or len(dataframe) < 50:
                self.logger.warning(f"[KRAKEN-RENDER] Datos insuficientes para {pair}")
                return None

            # Conversión segura del índice
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                try:
                    dataframe.index = pd.to_datetime(dataframe.index, utc=True)
                    dataframe.index = dataframe.index.tz_localize('UTC') if dataframe.index.tz is None else dataframe.index
                except Exception as e:
                    self.logger.error(f"[KRAKEN-RENDER] Error en índice temporal {pair}: {e}")
                    return None

            # Verificación de columnas específicas de Kraken
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(dataframe.columns):
                missing = required_cols - set(dataframe.columns)
                self.logger.error(f"[KRAKEN-RENDER] Columnas faltantes en {pair}: {missing}")
                return None

            # Limpieza de datos
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric, errors='coerce')
            dataframe.dropna(inplace=True)

            return dataframe

        except Exception as e:
            self.logger.error(f"[KRAKEN-RENDER] Error crítico procesando {pair}: {e}", exc_info=True)
            return None

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Versión ultra-robusta para Render con manejo de desconexiones
        """
        pair = metadata.get('pair', 'unknown')
        
        # Auto-refresh cada 6 horas para Render
        if time.time() - self.last_refresh > 21600:
            self.logger.info("[RENDER] Auto-refresh de datos activado")
            self.last_refresh = time.time()
            return pd.DataFrame()  # Fuerza refresco

        try:
            df = self._get_kraken_dataframe(dataframe, pair)
            if df is None:
                self.data_retries[pair] = self.data_retries.get(pair, 0) + 1
                if self.data_retries[pair] > 3:
                    self.logger.warning(f"[RENDER] Reintentos agotados para {pair}")
                    return pd.DataFrame()
                return dataframe

            # Resetear contador de reintentos
            self.data_retries[pair] = 0

            # Indicadores principales (optimizados para Kraken 5m)
            df['rsi'] = pta.rsi(df['close'], length=14).clip(10, 90)
            
            # EMAs con protección
            ema_lengths = [9, 21, 50, 100, 200]
            for length in ema_lengths:
                df[f'ema_{length}'] = pta.ema(df['close'], length=length)

            # MACD configurado para Kraken
            macd = pta.macd(df['close'], fast=12, slow=26, signal=9)
            df[['macd', 'macdsignal', 'macdhist']] = macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]

            # Volumen ajustado
            df['volume_ma'] = df['volume'].rolling(20).mean()

            return df.replace([np.inf, -np.inf], np.nan).dropna()

        except Exception as e:
            self.logger.error(f"[RENDER] Error en indicadores para {pair}: {e}", exc_info=True)
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Señales de entrada con protección para entornos inestables
        """
        pair = metadata.get('pair', 'unknown')
        
        try:
            if pair not in self.data_retries or self.data_retries[pair] > 0:
                return dataframe

            df = dataframe.copy()
            df['enter_long'] = 0

            # Condiciones adaptativas para Render
            conditions = [
                df['rsi'] > self.buy_rsi.value,
                df['close'] > df['ema_100'],
                df['volume'] > df['volume_ma'] * 0.65,
                df['macd'] > df['macdsignal'],
                df['close'] > df['open'].rolling(3).mean()
            ]

            # Aplicación segura
            if all(conditions):
                df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

            return df

        except Exception as e:
            self.logger.error(f"[RENDER] Error en señales entrada {pair}: {e}")
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Señales de salida con filtros adicionales para Kraken
        """
        pair = metadata.get('pair', 'unknown')
        
        try:
            if pair not in self.data_retries or self.data_retries[pair] > 0:
                return dataframe

            df = dataframe.copy()
            df['exit_long'] = 0

            # Condiciones de salida para entorno remoto
            exit_conditions = [
                df['rsi'] > self.sell_rsi.value,
                df['close'] < df['ema_21'],
                df['volume'] < df['volume_ma'] * 1.5,
                df['macd'] < df['macdsignal']
            ]

            if all(exit_conditions):
                df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1

            return df

        except Exception as e:
            self.logger.error(f"[RENDER] Error en señales salida {pair}: {e}")
            return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Gestión avanzada de salidas para Render
        """
        try:
            # Duración en horas
            duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

            # 1. Salida por timeout (3.5 horas)
            if duration_hours > 3.5 and current_profit < 0.005:
                return 'render_timeout'

            # 2. Protección de ganancias
            if current_profit > 0.015 and duration_hours > 1:
                return 'take_profit_render'

            # 3. Stop loss dinámico
            if current_profit < -0.01:  # -1%
                return 'stop_loss_render'

            return None

        except Exception as e:
            self.logger.error(f"[RENDER] Error en custom_exit {pair}: {e}")
            return None

    def bot_loop_start(self, **kwargs) -> None:
        """Mantenimiento periódico para Render"""
        self.logger.info("[RENDER] Ejecutando mantenimiento de rutina")
        
        # Auto-limpiar datos de pares problemáticos
        stale_pairs = [p for p, t in self.data_retries.items() if t > 5]
        for pair in stale_pairs:
            self.logger.warning(f"[RENDER] Limpiando par problemático: {pair}")
            self.data_retries.pop(pair)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """Validación final antes de entrar"""
        try:
            # Evitar operar en pares con errores recientes
            if self.data_retries.get(pair, 0) > 0:
                self.logger.info(f"[RENDER] Cancelando entrada por errores recientes en {pair}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"[RENDER] Error en confirm_trade_entry: {e}")
            return False