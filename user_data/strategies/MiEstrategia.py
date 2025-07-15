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
    Estrategia Final de Microtrading para Kraken
    - Objetivo: 5-7.5% diario con $15 USD (3 trades de $5 c/u)
    - Timeframe: 3 minutos (alta frecuencia)
    - Stop Loss: 1.5% dinámico
    - Take Profit: 0.75%-1.0% escalonado
    - Cooldown: 45min por par tras pérdida / 1h global si drawdown >2%
    """

    # =============================================
    # 1. CONFIGURACIÓN BASE (RENDER + KRAKEN)
    # =============================================
    INTERFACE_VERSION = 3
    timeframe = '1m'  # Timeframe ultra-corto para más oportunidades
    can_short = False  # Solo operaciones largas
    process_only_new_candles = True  # Optimiza recursos en Render
    startup_candle_count = 50  # Velas iniciales para cálculos

    # =============================================
    # 2. GESTIÓN DE CAPITAL ($50 USD → 10 trades de $5)
    # =============================================
    stoploss = -0.015  # -1.5% (stop loss base)
    trailing_stop = True  # Trailing activo
    trailing_stop_positive = 0.005  # 0.5% (activación trailing)
    trailing_stop_positive_offset = 0.01  # 1.0% (inicio del trailing)

    # =============================================
    # 3. OBJETIVOS DE RENTABILIDAD (ROI escalonado)
    # =============================================
    minimal_roi = {
        "0": 0.005,  # 0.5% inmediato
        "3": 0.003,  # 0.3% a los 3 minutos
        "6": 0       # Cierre a los 6 minutos
    }

    # =============================================
    # 4. PARÁMETROS OPTIMIZADOS (3m timeframe)
    # =============================================
    buy_rsi = IntParameter(20, 35, default=28, space='buy')  # RSI para compras
    sell_rsi = IntParameter(65, 85, default=73, space='sell')  # RSI para ventas
    buy_volume = DecimalParameter(1.5, 3.0, decimals=1, default=2.0, space='buy')  # Filtro volumen

    # =============================================
    # 5. CONFIGURACIÓN DE ÓRDENES (KRAKEN)
    # =============================================
    order_types = {
        'entry': 'limit',  # Evita slippage
        'exit': 'limit',
        'stoploss': 'market',  # Stop loss como mercado
        'stoploss_on_exchange': True  # Crítico para ejecución en Kraken
    }
    order_time_in_force = {
        'entry': 'IOC',  # "Immediate-or-Cancel" para entradas rápidas
        'exit': 'GTC'  # "Good-Til-Canceled" para salidas
    }

    # =============================================
    # 6. INIT (INICIALIZACIÓN - ¡IMPORTANTE PARA RENDER!)
    # =============================================
    def __init__(self, config: Dict) -> None:
        super().__init__(config)  # Hereda configuración base de Freqtrade
        
        # A. Configuración de Logging
        self.logger = logging.getLogger(__name__)
        
        # B. Sistema de Cooldown Dinámico
        self.loss_timestamps = {}  # Registro de pérdidas por par: {"AVAX/USD": datetime}
        self.consecutive_losses = {}  # Contador de pérdidas consecutivas: {"SOL/USD": 2}
        self.global_cooldown_until = None  # Tiempo de cooldown global
        
        # C. Parámetros de Cooldown (en segundos)
        self.cooldown_config = {
            'after_loss': 45 * 60,  # 45 minutos de cooldown tras pérdida
            'consecutive_loss': 90 * 60,  # 90 minutos si hay 2+ pérdidas seguidas
            'global_drawdown': 0.02,  # Activa cooldown global si pérdida >2%
            'profit_threshold': 0.05  # Ganancias >5% reducen cooldown
        }
        
        # D. Conexión Segura a Kraken (con rate limit)
        self.kraken = ccxt.kraken({
            'enableRateLimit': True,  # Evita bans por exceso de requests
            'options': {'adjustForTimeDifference': True}  # Sincronización horaria
        })
        
        # E. Variables de Monitoreo
        self.last_analysis = datetime.now(timezone.utc)
        self.profit_accumulated = 0.0

    # =============================================
    # 7. INDICADORES TÉCNICOS (OPTIMIZADOS PARA 3m)
    # =============================================
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Calcula indicadores ultra-rápidos:
        - RSI(3): Sobrevcompra/venta a corto plazo
        - EMA(5): Tendencia inmediata
        - Estocástico(3,3): Confirmación adicional
        - Volumen Relativo: Filtro de liquidez
        """
        try:
            df = dataframe.copy()
            
            # A. RSI de 3 periodos (ultra-corto)
            df['rsi'] = pta.rsi(df['close'], length=2).clip(10, 90)  # Evita valores extremos
            
            # B. EMA rápida (5 velas)
            df['ema5'] = pta.ema(df['close'], length=5)
            
            # C. Estocástico rápido (3,3,3)
            stoch = pta.stoch(df['high'], df['low'], df['close'], k=3, d=3)
            df = pd.concat([df, stoch], axis=1)  # Une al DataFrame
            
            # D. Volumen Relativo vs. Media Móvil (10 velas)
            df['volume_ma10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma10']
            
            return df.dropna()  # Elimina velas con valores faltantes

        except Exception as e:
            self.logger.error(f"Error calculando indicadores: {e}", exc_info=True)
            return dataframe

    # =============================================
    # 8. SEÑALES DE ENTRADA (CON FILTROS DE COOLDOWN)
    # =============================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Genera señales de compra cuando:
        1. RSI < 28 (sobreventa a corto plazo)
        2. Precio > EMA5 (tendencia alcista inmediata)
        3. Volumen > 2x el promedio (confirmación)
        4. Estocástico < 30 (confirmación adicional)
        """
        try:
            df = dataframe.copy()
            df['enter_long'] = 0  # Inicializa columna
            
            # Condiciones de Entrada
            conditions = [
                df['rsi'] < self.buy_rsi.value,  # RSI bajo
                df['close'] > df['ema20'],  # Precio sobre EMA rápida
                df['volume_ratio'] > self.buy_volume.value,  # Volumen alto
                df['STOCHk_3_3_3'] < 30,  # Estocástico en zona de compra
                df['STOCHd_3_3_3'] < 30
            ]
            
            # Aplica condiciones si existen
            if conditions:
                combined_cond = reduce(lambda x, y: x & y, conditions)  # Operador AND
                df.loc[combined_cond, 'enter_long'] = 1  # Marca señales
            
            return df

        except Exception as e:
            self.logger.error(f"Error en señales de entrada: {e}")
            return dataframe

    # =============================================
    # 9. SEÑALES DE SALIDA (PROTECCIÓN DE GANANCIAS)
    # =============================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Genera señales de venta cuando:
        1. RSI > 73 (sobrecompra)
        2. Estocástico > 70 (confirmación)
        3. Timeout de 24 minutos (8 velas)
        """
        try:
            df = dataframe.copy()
            df['exit_long'] = 0  # Inicializa columna
            
            # Condiciones de Salida
            exit_conditions = [
                df['rsi'] > self.sell_rsi.value,  # RSI alto
                df['STOCHk_3_3_3'] > 70,  # Estocástico en zona de venta
                df['STOCHd_3_3_3'] > 70
            ]
            
            # Aplica condiciones si existen
            if exit_conditions:
                exit_combined = reduce(lambda x, y: x & y, exit_conditions)  # Operador AND
                df.loc[exit_combined, 'exit_long'] = 1  # Marca señales
            
            return df

        except Exception as e:
            self.logger.error(f"Error en señales de salida: {e}")
            return dataframe

    # =============================================
    # 10. GESTIÓN DINÁMICA DE SALIDAS (COOLDOWN)
    # =============================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Decide cuándo salir de un trade activo:
        - Take Profit: 1% si se alcanza rápido
        - Timeout: 24 minutos (8 velas de 3m)
        - Stop Loss: 1.5% (trailing)
        - Registra pérdidas para cooldown
        """
        try:
            # A. Salida por Timeout (24 minutos)
            duration = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration > 24:
                return 'timeout_24min'
                
            # B. Take Profit Rápido (1%)
            if current_profit > 0.01:
                # Reduce cooldown si hay ganancias >5%
                if current_profit > self.cooldown_config['profit_threshold']:
                    self._reduce_cooldown(pair)
                return 'take_profit_1%'
                
            # C. Registra Pérdidas para Cooldown
            if current_profit < -0.01:  # -1%
                self._register_loss(pair, current_time)
                return 'stop_loss_1.5%'
                
            return None

        except Exception as e:
            self.logger.error(f"Error en custom_exit: {e}")
            return None

    # =============================================
    # 11. VALIDACIÓN ANTES DE ENTRAR (COOLDOWN/SPREAD)
    # =============================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Verifica antes de entrar:
        1. Si el par está en cooldown
        2. Si el spread es aceptable (<0.15%)
        3. Límite de trades diarios (20/bot)
        """
        try:
            # A. Verifica Cooldown Global
            now = datetime.now(timezone.utc)
            if self.global_cooldown_until and now < self.global_cooldown_until:
                self.logger.info("Cooldown global activo (drawdown >2%)")
                return False
                
            # B. Verifica Cooldown por Par
            if self.check_cooldown(pair):
                self.logger.info(f"Cooldown activo para {pair}")
                return False
                
            # C. Verifica Spread (no operar si > 0.15%)
            ticker = self.kraken.fetch_ticker(pair)
            spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            if spread > 0.0015:
                self.logger.info(f"Spread alto ({spread:.2%}), omitiendo trade")
                return False
                
            # D. Límite de 20 trades/bot/día
            if self._count_today_trades() >= 20:
                self.logger.info("Límite diario alcanzado (20 trades)")
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error en confirm_trade_entry: {e}")
            return False

    # =============================================
    # 12. MONITOREO DIARIO (DRAWDOWN/GLOBAL COOLDOWN)
    # =============================================
    def bot_loop_start(self, **kwargs) -> None:
        """
        Tareas ejecutadas al inicio de cada iteración:
        1. Calcula drawdown global
        2. Activa cooldown si drawdown >2%
        3. Reinicia contadores diarios
        """
        try:
            # A. Calcula Drawdown
            current = self.wallets.get_total('USD')
            highest = max(self.wallets.get_total('USD'), current)
            drawdown = (highest - current) / highest if highest > 0 else 0
            
            # B. Cooldown Global si Drawdown >2%
            if drawdown > self.cooldown_config['global_drawdown']:
                self.global_cooldown_until = datetime.now(timezone.utc) + timedelta(hours=1)
                self.logger.warning(f"Cooldown global activado (drawdown: {drawdown:.2%})")
                
            # C. Reinicio Diario (00:00 UTC)
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < 5:  # Ejecución una vez al día
                self.loss_timestamps = {}
                self.consecutive_losses = {}
                self.logger.info("Contadores diarios reiniciados")

        except Exception as e:
            self.logger.error(f"Error en bot_loop_start: {e}")

    # =============================================
    # 13. FUNCIONES AUXILIARES (COOLDOWN)
    # =============================================
    def check_cooldown(self, pair: str) -> bool:
        """Verifica si un par está en cooldown"""
        last_loss = self.loss_timestamps.get(pair)
        if not last_loss:
            return False
            
        losses = self.consecutive_losses.get(pair, 0)
        cooldown = self.cooldown_config['consecutive_loss'] if losses >= 2 else self.cooldown_config['after_loss']
        
        return (datetime.now(timezone.utc) - last_loss).total_seconds() < cooldown

    def _register_loss(self, pair: str, loss_time: datetime) -> None:
        """Registra una pérdida y actualiza cooldown"""
        self.loss_timestamps[pair] = loss_time
        self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1

    def _reduce_cooldown(self, pair: str) -> None:
        """Reduce cooldown tras ganancias >5%"""
        if pair in self.consecutive_losses:
            self.consecutive_losses[pair] = max(0, self.consecutive_losses[pair] - 2)

    def _count_today_trades(self) -> int:
        """Cuenta trades ejecutados hoy"""
        today = datetime.now(timezone.utc).date()
        return len([t for t in self.trades if t.open_date_utc.date() == today])