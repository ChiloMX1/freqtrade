# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from freqtrade.strategy import timeframe_to_minutes
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as pta
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from functools import reduce
from freqtrade.strategy import IntParameter, CategoricalParameter, DecimalParameter
import logging

class MiEstrategia(IStrategy):
    """
    Estrategia optimizada para trading en vivo con:
    - Manejo robusto de errores
    - Protección contra datos incompletos
    - Logging detallado
    - Prevención de falsas señales
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.trailing_active = False
        self.trailing_roi = 0.004
        self.loss_timestamps = {}
        self.cooldowns = {}
        self._last_valid_index = None
        self._log = logging.getLogger(__name__)

    # Configuración base para trading en vivo
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    use_custom_stoploss = True
    use_custom_exit = True

    # Parámetros optimizados para live trading
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

    # Protección contra velas faltantes
    startup_candle_count: int = 210
    missing_candles_threshold = 0.05  # 5% de velas faltantes máximo

    # Parámetros optimizables
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Configuración de órdenes para live trading
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True
    }

    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc"
    }

    def _is_data_valid(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> bool:
        """
        Validación exhaustiva para trading en vivo:
        1. Verifica estructura básica
        2. Comprueba integridad temporal
        3. Valida que no haya gaps excesivos
        """
        if dataframe.empty:
            self._log.warning(f"[LIVE] DataFrame vacío recibido para {metadata['pair']}")
            return False

        # Verificación de columnas OHLCV
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not ohlcv_cols.issubset(dataframe.columns):
            missing = ohlcv_cols - set(dataframe.columns)
            self._log.error(f"[LIVE] Columnas OHLCV faltantes en {metadata['pair']}: {missing}")
            return False

        # Verificación de índice temporal
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            self._log.error(f"[LIVE] Índice no es DatetimeIndex en {metadata['pair']}")
            return False

        # Verificación de monotonicidad
        if not dataframe.index.is_monotonic_increasing:
            self._log.error(f"[LIVE] Índice no es monotónico creciente en {metadata['pair']}")
            return False

        # Detección de gaps temporales
        time_diff = dataframe.index.to_series().diff().dt.total_seconds()
        expected_diff = timeframe_to_minutes(self.timeframe) * 60
        gap_ratio = (time_diff > expected_diff * 1.5).mean()

        if gap_ratio > self.missing_candles_threshold:
            self._log.warning(f"[LIVE] Demasiados gaps temporales en {metadata['pair']}: {gap_ratio:.2%}")
            return False

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Calcula indicadores con protección extra para live trading:
        - Manejo de errores por indicador
        - Validación de valores extremos
        - Protección contra NaN/Inf
        """
        try:
            if not self._is_data_valid(dataframe, metadata):
                return dataframe

            df = dataframe.copy()

            # RSI con protección
            df['rsi'] = pta.rsi(df['close'], length=14).clip(0, 100)

            # EMAs esenciales
            for length in [9, 21, 50, 200]:
                df[f'ema_{length}'] = pta.ema(df['close'], length=length)

            # MACD con validación
            macd = pta.macd(df['close'])
            if not macd.empty:
                df[['macd', 'macdsignal', 'macdhist']] = macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]

            # Bollinger Bands
            bb = pta.bbands(df['close'], length=20, std=2)
            if not bb.empty:
                df[['bb_upperband', 'bb_middleband', 'bb_lowerband']] = bb[['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']]

            # Limpieza final para live trading
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.ffill(inplace=True)
            df.dropna(inplace=True)

            return df

        except Exception as e:
            self._log.error(f"[LIVE] Error crítico en indicadores para {metadata['pair']}: {str(e)}", exc_info=True)
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Generación de señales de entrada con protecciones para live trading:
        - Validación de volumen
        - Confirmación de tendencia
        - Filtrado de falsos positivos
        """
        try:
            if not self._is_data_valid(dataframe, metadata):
                return dataframe

            df = dataframe.copy()
            df['enter_long'] = 0

            # Condiciones base con protección
            conditions = [
                df['volume'] > df['volume'].rolling(20).mean() * 0.8,
                df['close'] > df['ema_200'],
                df['rsi'].between(30, 70),
                df['ema_9'] > df['ema_21'],
                df['close'] > df['open']
            ]

            # Aplicación segura de condiciones
            if conditions:
                enter_mask = reduce(lambda x, y: x & y, conditions)
                df.loc[enter_mask, 'enter_long'] = 1

            # Filtrado adicional para live
            df['enter_long'] = df['enter_long'].rolling(3, min_periods=1).max()

            return df

        except Exception as e:
            self._log.error(f"[LIVE] Error en señales de entrada para {metadata['pair']}: {str(e)}")
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Generación de señales de salida con protecciones para live trading:
        - Confirmación de reversión
        - Protección de ganancias
        - Stop dinámico
        """
        try:
            if not self._is_data_valid(dataframe, metadata):
                return dataframe

            df = dataframe.copy()
            df['exit_long'] = 0

            # Condiciones de salida
            exit_conditions = [
                df['rsi'] > self.sell_rsi.value,
                df['close'] < df['ema_9'],
                df['volume'] < df['volume'].rolling(20).mean() * 0.7
            ]

            if exit_conditions:
                exit_mask = reduce(lambda x, y: x & y, exit_conditions)
                df.loc[exit_mask, 'exit_long'] = 1

            return df

        except Exception as e:
            self._log.error(f"[LIVE] Error en señales de salida para {metadata['pair']}: {str(e)}")
            return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Lógica de salida personalizada para live trading:
        - Timeout por inactividad
        - Trailing stop dinámico
        - Protección contra pérdidas
        """
        try:
            # Validación de datos de entrada
            if not isinstance(trade, Trade) or current_time.tzinfo is None:
                return None

            # Cálculo seguro de duración
            duration = (current_time - trade.open_date_utc.replace(tzinfo=timezone.utc)).total_seconds() / 60

            # Salida por timeout (3 horas sin ganancia mínima)
            if duration > 180 and current_profit < 0.002:
                return "exit_timeout"

            # Activación de trailing dinámico
            if current_profit > 0.004:
                self.trailing_active = True
                self.trailing_roi = max(self.trailing_roi, 0.0025)

            # Registro de pérdidas para cooldown
            if current_profit < -0.005:  # -0.5%
                self.loss_timestamps[pair] = current_time
                return "exit_stoploss"

            return None

        except Exception as e:
            self._log.error(f"[LIVE] Error en custom_exit para {pair}: {str(e)}")
            return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stop loss dinámico para live trading:
        - Ajuste basado en volatilidad
        - Protección de ganancias
        """
        try:
            # Stop loss base
            stoploss = self.stoploss

            # Ajuste por volatilidad (si los datos están disponibles)
            if hasattr(self, 'data_porvider'):
                candles = self.dp.get_pair_dataframe(pair, self.timeframe)
                if len(candles) > 20:
                    atr = pta.atr(candles['high'], candles['low'], candles['close'], length=14).iloc[-1]
                    stoploss = max(stoploss, -2 * atr / current_rate)

            # Protección de ganancias
            if current_profit > 0.01:  # +1%
                stoploss = max(stoploss, -0.005)  # No permitir perder más de 0.5%

            return stoploss

        except Exception as e:
            self._log.error(f"[LIVE] Error en custom_stoploss para {pair}: {str(e)}")
            return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Confirmación adicional antes de entrar en trade (live trading)
        """
        try:
            # Verificar cooldown
            if self.cooldown_active(pair, datetime.now(timezone.utc)):
                self._log.info(f"[LIVE] Cooldown activo para {pair}")
                return False

            # Verificar volumen reciente
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 3:
                return False

            last_candle = dataframe.iloc[-1]
            if last_candle['volume'] < dataframe['volume'].rolling(20).mean().iloc[-1] * 0.5:
                self._log.info(f"[LIVE] Volumen insuficiente para {pair}")
                return False

            return True

        except Exception as e:
            self._log.error(f"[LIVE] Error en confirm_trade_entry para {pair}: {str(e)}")
            return False

    def cooldown_active(self, pair: str, current_time: datetime) -> bool:
        """
        Verificación de cooldown para live trading
        """
        try:
            last_loss = self.loss_timestamps.get(pair)
            if last_loss and (current_time - last_loss) < timedelta(minutes=45):
                return True
            return False
        except Exception as e:
            self._log.error(f"[LIVE] Error en cooldown_active: {str(e)}")
            return False