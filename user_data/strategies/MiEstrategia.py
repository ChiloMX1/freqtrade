# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as pta
import talib.abstract as taba
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import reduce
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy import BooleanParameter, DecimalParameter
from freqtrade.strategy import informative
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.strategy import IntParameter

from freqtrade.strategy import (
    merge_informative_pair,
    stoploss_from_open,
    IntParameter,
    DecimalParameter,
    BooleanParameter,
)

def crossed_above(series1, series2):
    if not hasattr(series1, "shift"):
        return False
    if isinstance(series2, (int, float)):
        return (series1.shift(1) < series2) & (series1 > series2)
    else:
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)

class MiEstrategia(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # 📌 Para trailing dinámico
        self.trailing_active = False      # Activado cuando ROI > 0.004
        self.trailing_roi = 0.004         # ROI base mínimo para activar trailing

        # 📌 Cooldown por pérdida
        self.loss_timestamps = {}  # Diccionario para guardar últimos trades negativos por par

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False

    # ROI mínimo por trade: microganancias escalables
    minimal_roi = {
        "40": 0.003,   # 0.3%
        "20": 0.005,   # 0.5%
        "0": 0.065     # 6.5% fallback
    }

    # Desactivar uso de parámetros heredados
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    # Stoploss ajustado a -1%
    stoploss = -0.01

    # Trailing stop activo para asegurar ganancias pequeñas
    trailing_stop = True
    trailing_stop_positive = 0.002     # Activa trailing a partir de 0.2%
    trailing_stop_positive_offset = 0.004  # Sigue ganando hasta 0.4%
    trailing_only_offset_is_reached = True

    # Comportamiento general
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_exit = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    
    startup_candle_count: int = 210

    # RSI personalizado
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Tipos de orden ajustados para velocidad
    order_types = {
        "entry": "market",
        "exit": "market",     # ⚠ ahora usamos 'market' para evitar pérdidas por timeout
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {"tema": {}},
            "subplots": {
                "MACD": {"macd": {"color": "blue"}, "macdsignal": {"color": "orange"}},
                "RSI": {"rsi": {"color": "red"}}
            }
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)
        dataframe["ema50"] = pta.ema(dataframe["close"], length=50)
        dataframe["tema"] = pta.ema(dataframe["close"], length=9)

        macd = pta.macd(dataframe["close"])
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            dataframe["macd"] = macd["MACD_12_26_9"]
            dataframe["macdsignal"] = macd["MACDs_12_26_9"]
            dataframe["macdhist"] = macd["MACDh_12_26_9"]



        dataframe["mfi"] = pta.mfi(
            high=dataframe["high"].astype(float),
            low=dataframe["low"].astype(float),
            close=dataframe["close"].astype(float),
            volume=dataframe["volume"].astype(float)
        )

        dataframe["volume_mean"] = dataframe["volume"].rolling(window=24).mean()

        bbands = pta.bbands(dataframe["close"], length=20, std=2)
        if not bbands.empty:
            dataframe["bb_lowerband"] = bbands["BBL_20_2.0"]
            dataframe["bb_middleband"] = bbands["BBM_20_2.0"]
            dataframe["bb_upperband"] = bbands["BBU_20_2.0"]
            dataframe["bb_percent"] = (
                (dataframe["close"] - dataframe["bb_lowerband"]) /
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
            )
            dataframe["bb_width"] = (
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) /
                dataframe["bb_middleband"]
            )

        dataframe['ema_9'] = pta.ema(dataframe, timeperiod=9)
        dataframe['ema_21'] = pta.ema(dataframe, timeperiod=21)
        dataframe['ema_200'] = pta.ema(dataframe, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe.copy()
        dataframe.dropna(inplace=True)

        # ⚠️ Asegúrate de haber generado las EMAs y MACD en populate_indicators

        # Verificar cooldown por pérdida previa
        last_loss_time = self.loss_timestamps.get(metadata['pair'])
        if last_loss_time:
            cooldown_minutes = 45
            minutes_since_loss = (datetime.now(timezone.utc) - last_loss_time).total_seconds() / 60
            if minutes_since_loss < cooldown_minutes:
                return dataframe  # ⛔ Evita entrada si aún está en cooldown

        conditions = []

        # 👇 Ejemplo original (conserva lo que tú ya tenías)
        conditions.append(dataframe['rsi'] > 50)
        conditions.append(dataframe['volume'] > 0)

        # ✅ Punto 1: Evitar entradas en tendencia bajista inmediata
        # Confirmar EMA 9 > EMA 21
        conditions.append(dataframe['ema_9'] > dataframe['ema_21'])

        # ✅ Punto 2: Filtro de volatilidad
        # Validar que la vela actual no tenga un spike fuera de lo normal
        volatility = (dataframe['high'] - dataframe['low']).rolling(window=5).mean()
        conditions.append((dataframe['high'] - dataframe['low']) < volatility)

        # ✅ Punto 3: Filtro de volumen decreciente
        # Evita entradas si el volumen viene cayendo por 3 velas consecutivas
        conditions.append(~((dataframe['volume'] > dataframe['volume'].shift(1)) &
                            (dataframe['volume'].shift(1) > dataframe['volume'].shift(2))))

        # ✅ Punto 5: Confirmación con soporte/resistencia (EMA200)
        # Solo entrar si el precio actual está por encima de EMA200
        conditions.append(dataframe['close'] > dataframe['ema_200'])

        # 📌 Verificar cooldown por pérdida anterior
        row = dataframe.iloc[-1]  # ✅ Corregido: se define row antes de usarlo
        if self.cooldown_active(metadata['pair'], metadata['datetime']):
            dataframe.loc[row.index, 'enter_long'] = 0
            return dataframe

        # ✅ Punto 7: Cooldown por pérdidas anteriores
        pair = metadata['pair']
        now = datetime.utcnow()

        if pair in self.cooldowns and now < self.cooldowns[pair]:
            return dataframe

        if pair in self.trade_data and self.trade_data[pair]['last_result'] == 'loss':
            self.cooldowns[pair] = now + timedelta(minutes=45)

        # 👇 Aquí se aplican las condiciones como ya estaba en tu código original
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (crossed_above(dataframe["rsi"], self.sell_rsi.value)) &
                (dataframe["tema"] > dataframe["bb_middleband"]) &
                (dataframe["tema"] < dataframe["tema"].shift(1)) &
                (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        Exit trade si:
        - Supera 180 minutos sin alcanzar 0.2% de ROI
        - ROI supera 0.4% y se activa trailing dinámico
        """
        # Obtener duración del trade en minutos
        duration = (current_time - trade.open_date_utc).total_seconds() / 60

        # Cierre por duración sin rendimiento
        if duration > 180 and current_profit < 0.002:
            return "timeout_exit"

        # Activar trailing dinámico si ROI supera 0.4%
        if current_profit > 0.004:
            self.trailing_active = True
            self.trailing_roi = 0.0025  # Subir trailing para asegurar ganancia

        # Registrar pérdida si el trade cerró en negativo
        if current_profit < 0:
            self.loss_timestamps[pair] = current_time

        return None  # No salir si no se cumple ninguna condición
    
    def cooldown_active(self, pair: str, current_time: datetime) -> bool:
        """
        Retorna True si el par está dentro del periodo de cooldown de 45 minutos.
        """
        last_loss_time = self.loss_timestamps.get(pair)
        if last_loss_time:
            cooldown_duration = timedelta(minutes=45)
            if current_time - last_loss_time < cooldown_duration:
                return True
        return False
    

        