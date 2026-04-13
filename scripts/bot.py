"""
Торговый бот XAUUSD для MetaTrader 5 (Windows).

Запуск в режиме наблюдения (без реальных ордеров):
    python scripts/bot.py --dry-run

Запуск в боевом режиме (DEMO счёт!):
    python scripts/bot.py

Параметры:
    --symbol      Символ (default: XAUUSD)
    --threshold   Порог вероятности сигнала (default: 0.70)
    --lot         Фиксированный лот (если задан, --risk игнорируется)
    --risk        Риск на сделку в % от баланса (default: 1.0)
    --tp          Тейк-профит в $ (default: 0.5)
    --sl          Стоп-лосс в $ (default: 0.5)
    --dry-run     Не открывать реальные ордера, только логировать
    --models-dir  Путь к папке models/ (default: ../models)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Путь к features.py (тот же каталог)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, MIN_BARS  # noqa: E402

# ── Настройка логирования
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("bot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD ML Bot")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--lot", type=float, default=None,
                   help="Фиксированный лот (если задан, --risk игнорируется)")
    p.add_argument("--risk", type=float, default=1.0,
                   help="Риск на сделку, %% от баланса (игнорируется если задан --lot)")
    p.add_argument("--tp-points", type=int, default=50,
                   help="Тейк-профит в пунктах (default: 50 = $0.50 для XAUUSD)")
    p.add_argument("--sl-points", type=int, default=50,
                   help="Стоп-лосс в пунктах (default: 50 = $0.50 для XAUUSD)")
    p.add_argument("--max-trades-day", type=int, default=1000,
                   help="Макс. сделок в день (default: 1000)")
    p.add_argument("--cooldown", type=int, default=30,
                   help="Секунд паузы после сделки (default: 30)")
    p.add_argument("--dry-run", action="store_true",
                   help="Режим наблюдения: сигналы есть, ордеров нет")
    p.add_argument("--models-dir", default=None,
                   help="Путь к папке models/")
    return p.parse_args()


# ── Загрузка модели

def load_model(models_dir: Path, tag: str):
    model_path = models_dir / f"lgbm_{tag}.joblib"
    meta_path = models_dir / f"meta_{tag}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ── Получение свечей из MT5

def get_bars(mt5, symbol: str, n: int = 200) -> pd.DataFrame:
    import MetaTrader5 as mt5lib
    rates = mt5lib.copy_rates_from_pos(symbol, mt5lib.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Нет данных MT5 для {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
    return df


# ── Расчёт размера лота

def calc_lot(mt5lib, symbol: str, sl_price: float, risk_pct: float) -> float:
    info = mt5lib.account_info()
    balance = info.balance
    sym_info = mt5lib.symbol_info(symbol)

    # Для XAUUSD: 1 лот = 100 унций, pip = 0.01
    # Стоимость 1 пипса на 1 лот = 0.01 * 100 = $1
    point = sym_info.point          # обычно 0.01 для XAUUSD
    lot_size = sym_info.trade_contract_size  # обычно 100

    sl_points = sl_price / point    # $0.50 / 0.01 = 50 points
    pip_value = point * lot_size    # $1.00 per lot per point

    risk_amount = balance * risk_pct / 100
    lot = risk_amount / (sl_points * pip_value)

    # Округляем до допустимого шага
    step = sym_info.volume_step     # обычно 0.01
    lot = round(lot / step) * step
    lot = max(sym_info.volume_min, min(sym_info.volume_max, lot))
    return lot


# ── Открытие ордера

def open_buy(mt5lib, symbol: str, lot: float, tp: float, sl: float,
             dry_run: bool) -> bool:
    ask = mt5lib.symbol_info_tick(symbol).ask
    sl_price = round(ask - sl, 2)
    tp_price = round(ask + tp, 2)

    if dry_run:
        log.info(f"[DRY-RUN] BUY {lot:.2f} лот @ {ask:.2f}  SL={sl_price}  TP={tp_price}")
        return True

    request = {
        "action": mt5lib.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5lib.ORDER_TYPE_BUY,
        "price": ask,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 20240001,
        "comment": "ml_bot",
        "type_time": mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    }
    result = mt5lib.order_send(request)
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"Ордер открыт: BUY {lot:.2f} @ {ask:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    else:
        log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
        return False


# ── Открытие SELL ордера

def open_sell(mt5lib, symbol: str, lot: float, tp: float, sl: float,
              dry_run: bool) -> bool:
    bid = mt5lib.symbol_info_tick(symbol).bid
    sl_price = round(bid + sl, 2)
    tp_price = round(bid - tp, 2)

    if dry_run:
        log.info(f"[DRY-RUN] SELL {lot:.2f} лот @ {bid:.2f}  SL={sl_price}  TP={tp_price}")
        return True

    request = {
        "action": mt5lib.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5lib.ORDER_TYPE_SELL,
        "price": bid,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 20240002,
        "comment": "ml_bot_short",
        "type_time": mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    }
    result = mt5lib.order_send(request)
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"Ордер открыт: SELL {lot:.2f} @ {bid:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    else:
        log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
        return False


# ── Проверка открытых позиций

def has_open_position(mt5lib, symbol: str) -> bool:
    positions = mt5lib.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0


# ── Основной цикл

def run(args: argparse.Namespace) -> None:
    import MetaTrader5 as mt5lib

    # Инициализация MT5
    if not mt5lib.initialize():
        log.error(f"MT5 не инициализирован: {mt5lib.last_error()}")
        sys.exit(1)
    log.info(f"MT5 подключён: {mt5lib.terminal_info().name}")
    log.info(f"Аккаунт: {mt5lib.account_info().login}  Баланс: ${mt5lib.account_info().balance:.2f}")

    # Конвертируем пункты → доллары через символьную информацию MT5
    sym_info = mt5lib.symbol_info(args.symbol)
    point = sym_info.point  # для XAUUSD = 0.01
    exec_tp = round(args.tp_points * point, 2)  # TP для реального ордера
    exec_sl = round(args.sl_points * point, 2)  # SL для реального ордера
    log.info(f"TP: {args.tp_points} пунктов = ${exec_tp:.2f}  |  SL: {args.sl_points} пунктов = ${exec_sl:.2f}")

    # Загрузка моделей (тег всегда по обучающим параметрам tp=0.5 sl=0.5)
    models_dir = (
        Path(args.models_dir) if args.models_dir
        else Path(__file__).resolve().parent.parent / "models"
    )
    tag_long  = "M1_h5_long"
    tag_short = "M1_h5_short"

    model_long, meta_long = load_model(models_dir, tag_long)
    feature_cols = meta_long["feature_cols"]
    log.info(f"Модель LONG  загружена: {tag_long}   ROC-AUC={meta_long['roc_auc']}")

    model_short = None
    try:
        model_short, meta_short = load_model(models_dir, tag_short)
        log.info(f"Модель SHORT загружена: {tag_short}  ROC-AUC={meta_short['roc_auc']}")
    except FileNotFoundError:
        log.warning("Модель SHORT не найдена — торгуем только BUY")

    if args.lot is not None:
        log.info(f"Порог: {args.threshold}  TP={args.tp_points}п  SL={args.sl_points}п  Лот={args.lot} (фиксированный)")
    else:
        log.info(f"Порог: {args.threshold}  TP={args.tp_points}п  SL={args.sl_points}п  Риск={args.risk}%")
    log.info(f"Режим: {'DRY-RUN (без ордеров)' if args.dry_run else '⚠️  LIVE'}")
    log.info(f"Макс. сделок в день: {args.max_trades_day}  Cooldown: {args.cooldown} сек")
    log.info("─" * 60)

    trades_today = 0
    last_trade_time = None
    last_day = None

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Сброс счётчика в новый день
            if last_day != now.date():
                if last_day is not None:
                    log.info(f"Новый день. Сделок вчера: {trades_today}")
                trades_today = 0
                last_day = now.date()

            # Ждём начала новой минуты (первые 5 секунд)
            if now.second > 5:
                wait = 60 - now.second + 2
                time.sleep(wait)
                continue

            # Получаем свечи (250 = достаточно для SMA200 + запас)
            df = get_bars(mt5lib, args.symbol, n=250)

            # Считаем признаки
            df = build_features(df)
            last = df.dropna().iloc[[-1]]  # последний полный бар

            if last.empty:
                log.warning("Не хватает данных для признаков")
                time.sleep(10)
                continue

            # Предсказание
            X = last[feature_cols]
            prob_long  = model_long.predict_proba(X)[0, 1]
            prob_short = model_short.predict_proba(X)[0, 1] if model_short else 0.0

            signal_long  = prob_long  >= args.threshold
            signal_short = prob_short >= args.threshold

            # Фильтр по тренду: BUY только выше SMA200, SELL только ниже SMA200
            sma200 = df["close"].rolling(200).mean().iloc[-1]
            price_now = last["close"].iloc[-1]

            if np.isnan(sma200):
                log.warning("SMA200 = NaN (мало баров) — фильтр тренда отключён")
            else:
                trend_up   = price_now > sma200
                trend_down = price_now < sma200
                if signal_long and not trend_up:
                    log.info(f"  BUY заблокирован — цена ниже SMA200 ({sma200:.2f}), тренд вниз")
                    signal_long = False
                if signal_short and not trend_down:
                    log.info(f"  SELL заблокирован — цена выше SMA200 ({sma200:.2f}), тренд вверх")
                    signal_short = False

            bar_time = last.index[-1].strftime("%Y-%m-%d %H:%M")
            price = last["close"].iloc[-1]
            positions = mt5lib.positions_get(symbol=args.symbol)
            pos_info = f"позиций: {len(positions)}" if positions else "позиций: 0"

            signal_str = ""
            if signal_long:
                signal_str = "  >>> BUY <<<"
            elif signal_short:
                signal_str = "  >>> SELL <<<"

            log.info(
                f"[{bar_time}]  цена={price:.2f}"
                f"  long={prob_long:.3f}  short={prob_short:.3f}"
                f"  {pos_info}{signal_str}"
            )

            if has_open_position(mt5lib, args.symbol):
                if signal_long or signal_short:
                    log.info("Позиция уже открыта — пропускаем")
            elif trades_today >= args.max_trades_day:
                if signal_long or signal_short:
                    log.info(f"Лимит сделок на день ({args.max_trades_day}) достигнут — пропускаем")
            elif last_trade_time and (now - last_trade_time).seconds < args.cooldown:
                remaining = args.cooldown - (now - last_trade_time).seconds
                if signal_long or signal_short:
                    log.info(f"Cooldown — ждём ещё {remaining} сек")
            else:
                if args.lot is not None:
                    lot = args.lot
                else:
                    lot = calc_lot(mt5lib, args.symbol, exec_sl, args.risk)

                opened = False
                if signal_long:
                    opened = open_buy(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)
                elif signal_short:
                    opened = open_sell(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)

                if opened:
                    trades_today += 1
                    last_trade_time = now
                    log.info(f"Сделок сегодня: {trades_today}/{args.max_trades_day}")

            # Пауза до следующей минуты
            time.sleep(52)

        except KeyboardInterrupt:
            log.info("Остановка бота (Ctrl+C)")
            break
        except Exception as e:
            log.error(f"Ошибка: {e}", exc_info=True)
            time.sleep(15)

    mt5lib.shutdown()
    log.info("MT5 отключён")


if __name__ == "__main__":
    args = parse_args()
    run(args)
