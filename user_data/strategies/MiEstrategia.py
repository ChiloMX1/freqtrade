# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import pandas_ta as ta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
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

        # 🔁 Para trailing dinámico
        self.trailing_active = False
        self.trailing_roi = 0.004  # ROI base al iniciar trade

        # 🔁 Para cooldown por pérdida por par
        self.loss_timestamps = {}  # Guarda el último trade perdedor por par

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

    "protections": [
      {
        "method": "MaxTradeDuration",
        "stop_duration": 10800  // 3 horas en segundos

        "method": "CooldownPerPair",
        "duration": 45,
        "stop_duration": 0,
        "only_per_pair": true,
        "exit_reason": "stop_loss"
    }
]
      


    # Trailing stop activo para asegurar ganancias pequeñas
    "trailing_stop": true,
    "trailing_stop_positive": 0.002,
    "trailing_stop_positive_offset": 0.004,
    "trailing_only_offset_is_reached": true,
    "trailing_stop_dynamic": {
        "enabled": true,
        "trigger_threshold": 0.004,         // 0.4%
        "new_stop": 0.0025                  // cuando se dispare, usa este nuevo trailing
    }

    # Comportamiento general
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

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

        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)
        dataframe["ema50"] = ta.ema(dataframe["close"], length=50)
        dataframe["tema"] = ta.tema(dataframe["close"], length=9)

        macd = ta.macd(dataframe["close"])
        if not macd.empty and macd.shape[1] >= 3:
            dataframe["macd"] = macd.iloc[:, 0]
            dataframe["macdsignal"] = macd.iloc[:, 1]
            dataframe["macdhist"] = macd.iloc[:, 2]

        dataframe["mfi"] = ta.mfi(
            high=dataframe["high"].astype(float),
            low=dataframe["low"].astype(float),
            close=dataframe["close"].astype(float),
            volume=dataframe["volume"].astype(float)
        )

        dataframe["volume_mean"] = dataframe["volume"].rolling(window=24).mean()

        bbands = ta.bbands(dataframe["close"], length=20, std=2)
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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    pair = metadata['pair']
    current_time = dataframe.index[-1]

    # ✅ Validación de cooldown activo para el par
    if pair in self.cooldowns:
        cooldown_time = self.cooldowns[pair]
        if current_time < cooldown_time:
            # Saltar entradas si el par está en cooldown
            dataframe.loc[:, 'enter_long'] = 0
            return dataframe

    # ================================
    # CAMBIO 1: Validación de tendencia alcista con EMA 9 > EMA 21
    # ================================
    dataframe['ema9'] = ta.EMA(dataframe['close'], timeperiod=9)
    dataframe['ema21'] = ta.EMA(dataframe['close'], timeperiod=21)
    tendencia_alcista = dataframe['ema9'] > dataframe['ema21']

    # ================================
    # CAMBIO 2: Filtro de volatilidad - evitar spikes
    # ================================
    dataframe['rango'] = dataframe['high'] - dataframe['low']
    dataframe['rango_avg'] = dataframe['rango'].rolling(window=5).mean()
    filtro_volatilidad = dataframe['rango'] < dataframe['rango_avg']

    # ================================
    # CAMBIO 3: Filtro de volumen decreciente - evitar trampas
    # ================================
    vol = dataframe['volume']
    volumen_estable = ~((vol.shift(1) > vol.shift(2)) & (vol.shift(2) > vol.shift(3)))

    # ================================
    # CAMBIO 5: Validación de soporte con EMA 200 (15m)
    # ================================
    informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
    informative_15m['ema200'] = ta.EMA(informative_15m['close'], timeperiod=200)
    informative_15m = informative_15m[['ema200']]
    dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, '15m', ffill=True)
    ema200_validacion = dataframe['close'] > dataframe['ema200_15m']

    # ================================
    # CONDICIONES DE ENTRADA ORIGINALES + NUEVAS CONDICIONES
    # ================================
    dataframe.loc[
        (
            (crossed_above(dataframe['rsi'], self.buy_rsi.value)) &
            (dataframe['tema'] > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0) &
            tendencia_alcista &                   # CAMBIO 1
            filtro_volatilidad &                  # CAMBIO 2
            volumen_estable &                     # CAMBIO 3
            ema200_validacion                     # CAMBIO 5
        ),
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

        def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                current_profit: float, **kwargs) -> Optional[str]:
    """
    Cierre forzado si han pasado más de 180 minutos y el profit es menor a 0.2%.
    """
    # CAMBIO 4: Duración máxima por trade (180 min sin superar 0.2%)
    max_duration = timedelta(minutes=180)
    min_profit = 0.002  # 0.2% como decimal

    if (current_time - trade.open_date_utc) > max_duration and current_profit < min_profit:
        return 'timeout_exit'


    """
    Ajustes personalizados para salidas:
    - Paso 6: Trailing dinámico si ROI > 0.4%
    - Paso 7: Cooldown de 45min en pares con pérdida reciente
    """

def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
    """
    Exit logic personalizada basada en ROI dinámico.
    """

    # ✅ Cambio aplicado según regla #6: Trailing dinámico
    if current_profit > 0.004:  # 0.4%
        self.trailing_stop_positive = 0.0025

    if current_profit < 0:  # Trade con pérdida
    self.cooldowns[pair] = current_time + timedelta(minutes=45)

    return None
