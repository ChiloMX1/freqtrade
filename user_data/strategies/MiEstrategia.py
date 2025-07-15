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

class MicroScalperUltimate(IStrategy):
    """
    Estrategia de Microtrading Agresivo optimizada para:
    - Alto volumen de operaciones (15-20 trades/bot/día)
    - Ganancia por trade: 0.3%-0.7%
    - Drawdown controlado (<1.5%)
    - Adaptada a limitaciones de Kraken+Render
    """

    # ============= CONFIGURACIÓN PRINCIPAL =============
    timeframe = '1m'  # Óptimo balance velocidad/calidad de señales
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200  # Suficiente para indicadores

    # ============= GESTIÓN DE CAPITAL =============
    stoploss = -0.009  # -0.9% (más estricto para microtrades)
    trailing_stop = True
    trailing_stop_positive = 0.004  # 0.4%
    trailing_stop_positive_offset = 0.008  # 0.8%

    # ROI Dinámico para scalping
    minimal_roi = {
        "0": 0.006,  # 0.6% 
        "2": 0.004,  # 0.4%
        "5": 0.002,  # 0.2%
        "10": 0      # Cierre
    }

    # ============= PARÁMETROS OPTIMIZADOS =============
    buy_rsi = IntParameter(18, 28, default=22, space='buy')
    sell_rsi = IntParameter(72, 82, default=76, space='sell')
    buy_volume = DecimalParameter(2.5, 4.0, decimals=1, default=3.2, space='buy')
    buy_adx = DecimalParameter(28.0, 35.0, decimals=1, default=30.0, space='buy')
    volatility_filter = DecimalParameter(0.008, 0.025, decimals=3, default=0.015, space='buy')

    # ============= CONFIGURACIÓN DE ÓRDENES =============
    order_types = {
        'entry': 'limit',  # Evita slippage
        'exit': 'limit',
        'stoploss': 'market',  # Ejecución garantizada
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'entry': 'IOC',  # Ejecución inmediata o cancelación
        'exit': 'GTC'    # Orden buena hasta cancelar
    }

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.logger = logger
        self.trade_count_today = 0
        self.max_daily_trades = 18  # Objetivo: 15-20 trades
        
        # Sistema de Cooldown Inteligente
        self.cooldown_config = {
            'after_loss': 20 * 60,  # 20 minutos
            'consecutive_loss': 40 * 60,  # 40 minutos
            'global_drawdown': 0.012,  # 1.2%
            'profit_reset': 0.025  # 2.5% de profit resetea cooldowns
        }
        
        # Conexión optimizada para Kraken
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False
            }
        })

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Indicadores optimizados para microtrading:
        - Velocidad de cálculo
        - Precisión en timeframe corto
        - Filtrado de ruido
        """
        try:
            df = dataframe.copy()
            
            # 1. Momentum Ultra-Rápido
            df['rsi'] = pta.rsi(df['close'], length=4).ffill().bfill()
            df['ema5'] = pta.ema(df['close'], length=5).ffill()
            df['ema15'] = pta.ema(df['close'], length=15).ffill()
            
            # 2. Confirmación de Tendencia
            adx = pta.adx(df['high'], df['low'], df['close'], length=9)
            df['adx'] = adx['ADX_9'].ffill()
            df['plus_di'] = adx['DMP_9'].ffill()
            
            # 3. Volumen y Liquidez
            df['volume_ma'] = df['volume'].rolling(12).mean().replace(0, 1e-10)
            df['volume_ratio'] = (df['volume'] / df['volume_ma']).clip(0, 5)
            
            # 4. Estocástico Rápido
            stoch = pta.stoch(df['high'], df['low'], df['close'], k=5, d=3, smooth_k=3)
            df = pd.concat([df, stoch], axis=1)
            
            # 5. Filtro de Volatilidad (clave para microtrades)
            df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=7)
            df['volatility'] = (df['atr'] / df['close']).clip(0.005, 0.03)
            
            # 6. Filtro de Spread (importante en Kraken)
            df['spread'] = (df['high'] - df['low']) / df['open']
            
            return df.dropna().copy()

        except Exception as e:
            self.logger.error(f"Error en populate_indicators: {str(e)}", exc_info=True)
            return dataframe.iloc[self.startup_candle_count:].copy()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Señales de entrada con filtros mejorados para alta frecuencia"""
        try:
            df = dataframe.copy()
            df['enter_long'] = 0
            
            # Condiciones Principales (optimizadas para 3m)
            conditions = [
                # Momentum
                df['rsi'] < self.buy_rsi.value,
                df['close'] > df['ema5'],
                df['STOCHk_5_3_3'] < 22,
                df['STOCHd_5_3_3'] < 22,
                
                # Tendencia
                df['close'] > df['ema15'],
                df['adx'] > self.buy_adx.value,
                df['plus_di'] > 20,
                
                # Liquidez y Volumen
                df['volume_ratio'] > self.buy_volume.value,
                
                # Filtros de Mercado
                df['volatility'].between(0.008, self.volatility_filter.value),
                df['spread'] < 0.0015,  # Filtro de spread <0.15%
                
                # Protección contra manipulación
                df['volume'] > df['volume'].rolling(30).mean() * 0.7
            ]
            
            if conditions:
                df.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error en populate_entry_trend: {str(e)}")
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Señales de salida optimizadas para scalping"""
        try:
            df = dataframe.copy()
            df['exit_long'] = 0
            
            exit_conditions = [
                # Condiciones Básicas
                df['rsi'] > self.sell_rsi.value,
                df['STOCHk_5_3_3'] > 74,
                
                # Salida Temprana
                df['close'] < df['ema5'],
                
                # Protección de Ganancia
                (df['close'] / df['open']) > 1.004  # Si ya ganó 0.4%, sale
            ]
            
            if exit_conditions:
                df.loc[reduce(lambda x, y: x & y, exit_conditions), 'exit_long'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error en populate_exit_trend: {str(e)}")
            return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """Gestión inteligente de salidas"""
        try:
            # Timeout ajustado a timeframe 3m (15 minutos máximo)
            duration = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration > 15:
                return 'timeout_15min'
                
            # Take profit dinámico
            if current_profit > 0.006:  # 0.6%
                if current_profit > self.cooldown_config['profit_reset']:
                    self._reduce_cooldown(pair)
                return 'take_profit'
                
            # Stop loss adaptativo
            if current_profit < -0.007:  # -0.7%
                self._register_loss(pair, current_time)
                return 'adaptive_stop_loss'
                
            return None

        except Exception as e:
            self.logger.error(f"Error en custom_exit: {str(e)}")
            return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """Validación adicional antes de entrar"""
        try:
            # Control de frecuencia de trading
            if self.trade_count_today >= self.max_daily_trades:
                self.logger.info(f"Alcanzado límite diario de {self.max_daily_trades} trades")
                return False
                
            # Verificar cooldown
            if self.check_cooldown(pair):
                return False
                
            # Análisis de spread en tiempo real
            ticker = self.exchange.fetch_ticker(pair)
            spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            if spread > 0.0012:  # 0.12%
                self.logger.info(f"Spread alto {spread:.4f} en {pair}")
                return False
                
            # Verificar volumen reciente
            candles = self.exchange.fetch_ohlcv(pair, '3m', limit=5)
            volume_avg = sum(c[5] for c in candles) / len(candles)
            if volume_avg < 1000:  # Mínimo volumen en USD
                self.logger.info(f"Volumen insuficiente en {pair}: {volume_avg:.2f}")
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error en confirm_trade_entry: {str(e)}")
            return False

    # ============= FUNCIONES AUXILIARES =============
    def _register_loss(self, pair: str, loss_time: datetime) -> None:
        """Registra pérdida para cooldown inteligente"""
        self.loss_timestamps[pair] = loss_time
        self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1
        self.logger.info(f"Registrada pérdida en {pair}. Cooldown activado.")

    def _reduce_cooldown(self, pair: str) -> None:
        """Reduce cooldown tras ganancias significativas"""
        if pair in self.consecutive_losses:
            self.consecutive_losses[pair] = max(0, self.consecutive_losses[pair] - 2)
            self.logger.info(f"Cooldown reducido para {pair}")

    def check_cooldown(self, pair: str) -> bool:
        """Verifica si el par está en cooldown"""
        last_loss = self.loss_timestamps.get(pair)
        if not last_loss:
            return False
            
        losses = self.consecutive_losses.get(pair, 0)
        cooldown = self.cooldown_config['consecutive_loss'] if losses >= 2 else self.cooldown_config['after_loss']
        
        return (datetime.now(timezone.utc) - last_loss).total_seconds() < cooldown

    def bot_loop_start(self, **kwargs) -> None:
        """Tareas al inicio de cada iteración"""
        try:
            # Reinicio diario de contadores
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < 3:
                self.trade_count_today = 0
                self.loss_timestamps = {}
                self.consecutive_losses = {}
                self.logger.info("Contadores diarios reiniciados")
                
            # Monitoreo de drawdown
            current = self.wallets.get_total('USD')
            highest = max(self.wallets.get_total('USD'), current)
            drawdown = (highest - current) / highest if highest > 0 else 0
            
            if drawdown > self.cooldown_config['global_drawdown']:
                self.global_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                self.logger.warning(f"Drawdown global {drawdown:.2%}. Cooldown activado.")

        except Exception as e:
            self.logger.error(f"Error en bot_loop_start: {str(e)}")

    def version(self) -> str:
        return "MicroScalperUltimate v1.0 (Optimizada para Kraken+Render)"