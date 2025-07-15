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
    Estrategia Final de Microtrading para Kraken (1m timeframe)
    - Objetivo: 5-7.5% diario con $15 USD (3 trades de $5 c/u)
    - Stop Loss: 1.5% dinámico
    - Take Profit: 0.75%-1.0% escalonado
    - Cooldown: 45min por par tras pérdida / 1h global si drawdown >2%
    """

    # =============================================
    # 1. CONFIGURACIÓN BASE (RENDER + KRAKEN)
    # =============================================
    INTERFACE_VERSION = 3
    timeframe = '1m'  # Cambiado a 1m para scalping
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 50  # Suficiente para EMA20 (20) + margen

    # =============================================
    # 2. GESTIÓN DE CAPITAL
    # =============================================
    stoploss = -0.015  # -1.5% (stop loss base)
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5% (activación trailing)
    trailing_stop_positive_offset = 0.01  # 1.0% (inicio del trailing)

    # =============================================
    # 3. OBJETIVOS DE RENTABILIDAD (ROI escalonado)
    # =============================================
    minimal_roi = {
        "0": 0.0075,  # 0.75% ROI inmediato
        "5": 0.005,   # 0.5% después de 5 velas
        "10": 0.003,  # 0.3% después de 10 velas
        "20": 0       # Cierre forzoso a las 20 velas
    }

    # =============================================
    # 4. PARÁMETROS OPTIMIZADOS (1m timeframe)
    # =============================================
    buy_rsi = IntParameter(20, 35, default=28, space='buy')
    sell_rsi = IntParameter(65, 85, default=73, space='sell')
    buy_volume = DecimalParameter(1.5, 3.0, decimals=1, default=2.0, space='buy')

    # =============================================
    # 5. CONFIGURACIÓN DE ÓRDENES (KRAKEN)
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
    # 6. INIT (INICIALIZACIÓN)
    # =============================================
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
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
        
        self.last_analysis = datetime.now(timezone.utc)
        self.profit_accumulated = 0.0

    # =============================================
    # 7. INDICADORES TÉCNICOS (OPTIMIZADOS PARA 1m)
    # =============================================
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Calcula indicadores ultra-rápidos para 1m:
        - RSI(2): Sensibilidad extrema
        - EMA(5) y EMA(20): Tendencia rápida y global
        - Estocástico(3,3): Confirmación
        - Volumen Relativo: Filtro de liquidez
        """
        try:
            df = dataframe.copy()
            
            # A. Indicadores de Momentum
            df['rsi'] = pta.rsi(df['close'], length=2).clip(10, 90)  # RSI ultra-corto
            df['ema5'] = pta.ema(df['close'], length=5)  # EMA rápida
            df['ema20'] = pta.ema(df['close'], length=20)  # EMA global
            
            # B. Estocástico Rápido
            stoch = pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            # C. Volumen Relativo
            df['volume_ma10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma10']
            
            return df.dropna()  # Elimina NaN para consistencia

        except Exception as e:
            self.logger.error(f"Error en populate_indicators: {e}", exc_info=True)
            return dataframe

    # =============================================
    # 8. SEÑALES DE ENTRADA (CON FILTROS)
    # =============================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Entradas cuando:
        1. RSI(2) < 28 (sobreventa extrema)
        2. Precio > EMA5 (tendencia inmediata alcista)
        3. Precio > EMA20 (tendencia global alcista)
        4. Volumen > 2x la media
        5. Estocástico K y D < 30
        """
        try:
            df = dataframe.copy()
            df['enter_long'] = 0
            
            conditions = [
                df['rsi'] < self.buy_rsi.value,
                df['close'] > df['ema5'],
                df['close'] > df['ema20'],
                df['volume_ratio'] > self.buy_volume.value,
                df['STOCHk_3_3_3'] < 30,
                df['STOCHd_3_3_3'] < 30
            ]
            
            if conditions:
                df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error en populate_entry_trend: {e}")
            return dataframe

    # =============================================
    # 9. SEÑALES DE SALIDA
    # =============================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Salidas cuando:
        1. RSI > 73 (sobrecompra)
        2. Estocástico K y D > 70
        """
        try:
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

        except Exception as e:
            self.logger.error(f"Error en populate_exit_trend: {e}")
            return dataframe

    # =============================================
    # 10. FUNCIONES AUXILIARES (COOLDOWN)
    # =============================================
    def check_cooldown(self, pair: str) -> bool:
        last_loss = self.loss_timestamps.get(pair)
        if not last_loss:
            return False
        losses = self.consecutive_losses.get(pair, 0)
        cooldown = self.cooldown_config['consecutive_loss'] if losses >= 2 else self.cooldown_config['after_loss']
        return (datetime.now(timezone.utc) - last_loss).total_seconds() < cooldown

    def _register_loss(self, pair: str, loss_time: datetime) -> None:
        self.loss_timestamps[pair] = loss_time
        self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1

    def _reduce_cooldown(self, pair: str) -> None:
        if pair in self.consecutive_losses:
            self.consecutive_losses[pair] = max(0, self.consecutive_losses[pair] - 2)

    def _count_today_trades(self) -> int:
        today = datetime.now(timezone.utc).date()
        return len([t for t in self.trades if t.open_date_utc.date() == today])

    # =============================================
    # 11. MÉTODOS HEREDADOS (CUSTOM_EXIT, ETC.)
    # =============================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Lógica personalizada de salida:
        - Take Profit: 1% si se alcanza rápido
        - Timeout: 24 minutos (8 velas de 3m original, ajustado a 12 velas de 1m)
        - Stop Loss: 1.5% (trailing)
        """
        try:
            # Timeout reducido para 1m (12 velas = 12 minutos)
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
            self.logger.error(f"Error en custom_exit: {e}")
            return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Valida antes de entrar:
        1. Cooldown activo
        2. Spread < 0.15%
        3. Límite de 20 trades/día
        """
        try:
            now = datetime.now(timezone.utc)
            if self.global_cooldown_until and now < self.global_cooldown_until:
                self.logger.info("Cooldown global activo (drawdown >2%)")
                return False
                
            if self.check_cooldown(pair):
                self.logger.info(f"Cooldown activo para {pair}")
                return False
                
            ticker = self.kraken.fetch_ticker(pair)
            spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            if spread > 0.0015:
                self.logger.info(f"Spread alto ({spread:.2%}), omitiendo trade")
                return False
                
            if self._count_today_trades() >= 20:
                self.logger.info("Límite diario alcanzado (20 trades)")
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error en confirm_trade_entry: {e}")
            return False

    def bot_loop_start(self, **kwargs) -> None:
        """
        Monitorea drawdown global y reinicia contadores diarios.
        """
        try:
            current = self.wallets.get_total('USD')
            highest = max(self.wallets.get_total('USD'), current)
            drawdown = (highest - current) / highest if highest > 0 else 0
            
            if drawdown > self.cooldown_config['global_drawdown']:
                self.global_cooldown_until = datetime.now(timezone.utc) + timedelta(hours=1)
                self.logger.warning(f"Cooldown global activado (drawdown: {drawdown:.2%})")
                
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < 5:
                self.loss_timestamps = {}
                self.consecutive_losses = {}
                self.logger.info("Contadores diarios reiniciados")

        except Exception as e:
            self.logger.error(f"Error en bot_loop_start: {e}")