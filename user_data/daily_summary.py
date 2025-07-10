import datetime
from freqtrade.rpc.telegram import send_msg
from freqtrade.rpc import rpc_manager
from freqtrade.persistence import Trade
from freqtrade.optimize.backtesting import Backtesting
from pandas import Timestamp
import schedule

def get_midnight_timestamp():
    """Retorna un timestamp desde las 00:00 del día actual (UTC)."""
    now = datetime.datetime.utcnow()
    return datetime.datetime(year=now.year, month=now.month, day=now.day)

def generate_summary(since: datetime.datetime):
    """Genera el resumen desde una fecha dada hasta ahora."""
    now = datetime.datetime.utcnow()
    trades = Trade.query \
        .filter(Trade.close_date >= since) \
        .filter(Trade.close_date <= now) \
        .all()

    total_profit = sum([t.close_profit for t in trades])
    wins = len([t for t in trades if t.close_profit > 0])
    losses = len([t for t in trades if t.close_profit <= 0])
    count = len(trades)

    msg = f"📊 *Resumen de operaciones desde {since.strftime('%H:%M')} UTC hasta ahora:*\n\n"
    msg += f"📈 Total de operaciones: *{count}*\n"
    msg += f"✅ Ganadoras: *{wins}*\n"
    msg += f"❌ Perdedoras: *{losses}*\n"
    msg += f"💰 Ganancia neta: *{total_profit:.2f} USD*\n"
    return msg

def resumen_manual(update, context):
    """Comando /resumen en Telegram."""
    start = get_midnight_timestamp()
    msg = generate_summary(start)
    send_msg(msg, parse_mode='Markdown')

def resumen_automatico():
    """Ejecutado automáticamente a las 00:00 UTC todos los días."""
    start = get_midnight_timestamp()
    msg = generate_summary(start)
    send_msg("📤 *Resumen diario automático:*\n\n" + msg, parse_mode='Markdown')

def schedule_summary():
    """Agrega la tarea automática a las 00:00 UTC."""
    schedule.every().day.at("00:00").do(resumen_automatico)

# REGISTRA EL COMANDO MANUAL AL CARGAR EL ARCHIVO
rpc_manager.register_rpc_command('resumen', resumen_manual)
