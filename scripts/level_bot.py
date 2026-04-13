"""
Торговый бот XAUUSD на основе УРОВНЕВЫХ моделей (level_*).

Особенности:
  - Загружает модели с префиксом level_ (level_lgbm_*, level_meta_*)
  - Запрашивает 1500 M1-баров (нужно для дневного уровня 1440 баров)
  - Те же параметры управления рисками что в bot.py

Запуск:
    python scripts/level_bot.py --dry-run
    python scripts/level_bot.py
    python scripts/level_bot.py --lot 1 --threshold 0.65
    python scripts/level_bot.py --tp-points 100 --sl-points 50 --risk 1.5

Параметры:
    --symbol        Символ (default: XAUUSD)
    --threshold     Порог вероятности сигнала (default: 0.65)
    --lot           Фиксированный лот (если задан, --risk игнорируется)
    --risk          Риск на сделку в %% от баланса (default: 1.0)
    --tp-points     Тейк-профит в пунктах (default: 100)
    --sl-points     Стоп-лосс в пунктах (default: 50)
    --max-trades-day Макс. сделок в день (default: 1000)
    --cooldown      Секунд паузы после сделки (default: 30)
    --dry-run       Не открывать реальные ордера
    --models-dir    Путь к папке models/
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

# Уровневые признаки (тот же файл что и при обучении)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level_features import build_all_features, MIN_BARS_LEVELS  # noqa: E402

BARS_TO_FETCH = 100   # 30 для уровней + 50 для SMA50 + запас

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"level_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"),
    ],
)
log = logging.getLogger("level_bot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD Level ML Bot")
    p.add_argument("--symbol",         default="XAUUSD")
    p.add_argument("--threshold",      type=float, default=0.65,
                   help="Порог вероятности (уровневые модели, default: 0.65)")
    p.add_argument("--lot",            type=float, default=None,
                   help="Фиксированный лот (если задан, --risk игнорируется)")
    p.add_argument("--risk",           type=float, default=1.0,
                   help="Риск на сделку, %% от баланса")
    p.add_argument("--tp-points",      type=int,   default=100,
                   help="Тейк-профит в пунктах (default: 100)")
    p.add_argument("--sl-points",      type=int,   default=50,
                   help="Стоп-лосс в пунктах (default: 50)")
    p.add_argument("--max-trades-day", type=int,   default=1000,
                   help="Макс. сделок в день (default: 1000)")
    p.add_argument("--cooldown",       type=int,   default=30,
                   help="Секунд паузы после сделки (default: 30)")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--models-dir",     default=None)
    return p.parse_args()


# ── Загрузка модели

def load_level_model(models_dir: Path, tag: str):
    model_path = models_dir / f"level_lgbm_{tag}.joblib"
    meta_path  = models_dir / f"level_meta_{tag}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ── Получение свечей из MT5

def get_bars(mt5lib, symbol: str, n: int) -> pd.DataFrame:
    rates = mt5lib.copy_rates_from_pos(symbol, mt5lib.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Нет данных MT5 для {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
    return df


# ── Расчёт лота по риску

def calc_lot(mt5lib, symbol: str, sl_points_dollar: float, risk_pct: float) -> float:
    info     = mt5lib.account_info()
    sym_info = mt5lib.symbol_info(symbol)
    balance  = info.balance
    point    = sym_info.point
    lot_size = sym_info.trade_contract_size

    sl_pts     = sl_points_dollar / point
    pip_value  = point * lot_size
    risk_amt   = balance * risk_pct / 100
    lot        = risk_amt / (sl_pts * pip_value)

    step = sym_info.volume_step
    lot  = round(lot / step) * step
    return max(sym_info.volume_min, min(sym_info.volume_max, lot))


# ── Открытие BUY

def open_buy(mt5lib, symbol: str, lot: float, tp: float, sl: float,
             dry_run: bool) -> bool:
    ask      = mt5lib.symbol_info_tick(symbol).ask
    sl_price = round(ask - sl, 2)
    tp_price = round(ask + tp, 2)

    if dry_run:
        log.info(f"[DRY-RUN] BUY {lot:.2f} лот @ {ask:.2f}  SL={sl_price}  TP={tp_price}")
        return True

    request = {
        "action":       mt5lib.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5lib.ORDER_TYPE_BUY,
        "price":        ask,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        20260010,   # magic = level_bot BUY
        "comment":      "level_bot",
        "type_time":    mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    }
    result = mt5lib.order_send(request)
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"Ордер: BUY {lot:.2f} @ {ask:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
    return False


# ── Открытие SELL

def open_sell(mt5lib, symbol: str, lot: float, tp: float, sl: float,
              dry_run: bool) -> bool:
    bid      = mt5lib.symbol_info_tick(symbol).bid
    sl_price = round(bid + sl, 2)
    tp_price = round(bid - tp, 2)

    if dry_run:
        log.info(f"[DRY-RUN] SELL {lot:.2f} лот @ {bid:.2f}  SL={sl_price}  TP={tp_price}")
        return True

    request = {
        "action":       mt5lib.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5lib.ORDER_TYPE_SELL,
        "price":        bid,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        20260011,   # magic = level_bot SELL
        "comment":      "level_bot_short",
        "type_time":    mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    }
    result = mt5lib.order_send(request)
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"Ордер: SELL {lot:.2f} @ {bid:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
    return False


def has_open_position(mt5lib, symbol: str) -> bool:
    positions = mt5lib.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0


# ── Форматирование уровневого контекста для лога

def level_context(last_row: pd.Series) -> str:
    parts = []
    for col in ["dist_day_high", "dist_day_low", "near_day_high", "near_day_low",
                "bounce_buy", "bounce_sell", "breakout_signal_up", "breakout_signal_down",
                "fakeout_signal_up", "fakeout_signal_down"]:
        if col in last_row.index:
            v = last_row[col]
            if v and v != 0:
                parts.append(f"{col}={v:.2f}" if isinstance(v, float) else f"{col}=1")
    return "  [" + "  ".join(parts) + "]" if parts else ""


# ── Основной цикл

def run(args: argparse.Namespace) -> None:
    import MetaTrader5 as mt5lib

    if not mt5lib.initialize():
        log.error(f"MT5 не инициализирован: {mt5lib.last_error()}")
        sys.exit(1)
    log.info(f"MT5 подключён: {mt5lib.terminal_info().name}")
    log.info(f"Аккаунт: {mt5lib.account_info().login}  Баланс: ${mt5lib.account_info().balance:.2f}")

    sym_info = mt5lib.symbol_info(args.symbol)
    point    = sym_info.point
    exec_tp  = round(args.tp_points * point, 2)
    exec_sl  = round(args.sl_points * point, 2)
    log.info(f"TP: {args.tp_points}п = ${exec_tp:.2f}  |  SL: {args.sl_points}п = ${exec_sl:.2f}")
    log.info(f"Запрашивается баров: {BARS_TO_FETCH}")

    models_dir = (
        Path(args.models_dir) if args.models_dir
        else Path(__file__).resolve().parent.parent / "models"
    )
    tag_long  = "M1_h3_long"
    tag_short = "M1_h3_short"

    model_long, meta_long = load_level_model(models_dir, tag_long)
    feature_cols = meta_long["feature_cols"]
    log.info(f"Модель LONG  загружена: level_{tag_long}  ROC-AUC={meta_long['roc_auc']}")

    model_short = None
    try:
        model_short, meta_short = load_level_model(models_dir, tag_short)
        log.info(f"Модель SHORT загружена: level_{tag_short}  ROC-AUC={meta_short['roc_auc']}")
    except FileNotFoundError:
        log.warning("Модель SHORT не найдена — торгуем только BUY")

    if args.lot is not None:
        log.info(f"Порог: {args.threshold}  TP={args.tp_points}п  SL={args.sl_points}п  Лот={args.lot} (фиксированный)")
    else:
        log.info(f"Порог: {args.threshold}  TP={args.tp_points}п  SL={args.sl_points}п  Риск={args.risk}%")
    log.info(f"Режим: {'DRY-RUN (без ордеров)' if args.dry_run else '⚠️  LIVE'}")
    log.info(f"Макс. сделок в день: {args.max_trades_day}  Cooldown: {args.cooldown} сек")
    log.info("─" * 60)

    trades_today    = 0
    last_trade_time = None
    last_day        = None

    while True:
        try:
            now = datetime.now(timezone.utc)

            if last_day != now.date():
                if last_day is not None:
                    log.info(f"Новый день. Сделок вчера: {trades_today}")
                trades_today = 0
                last_day = now.date()

            if now.second > 5:
                wait = 60 - now.second + 2
                time.sleep(wait)
                continue

            # Получаем 1500 баров — достаточно для дневных уровней (1440 M1)
            df = get_bars(mt5lib, args.symbol, n=BARS_TO_FETCH)

            # Строим ВСЕ признаки (базовые + уровневые)
            df = build_all_features(df)
            last = df.dropna(subset=feature_cols).iloc[[-1]]

            if last.empty:
                log.warning("Не хватает данных для признаков (нужно больше баров?)")
                time.sleep(10)
                continue

            last_row = last.iloc[0]
            X = last[feature_cols]

            prob_long  = model_long.predict_proba(X)[0, 1]
            prob_short = model_short.predict_proba(X)[0, 1] if model_short else 0.0

            signal_long  = prob_long  >= args.threshold
            signal_short = prob_short >= args.threshold
            price_now    = last_row["close"]

            # Лог текущего состояния + уровневый контекст
            bar_time   = last.index[-1].strftime("%Y-%m-%d %H:%M")
            positions  = mt5lib.positions_get(symbol=args.symbol)
            pos_info   = f"позиций: {len(positions)}" if positions else "позиций: 0"
            signal_str = ("  >>> BUY <<<" if signal_long
                          else "  >>> SELL <<<" if signal_short else "")
            lvl_ctx    = level_context(last_row)

            log.info(
                f"[{bar_time}]  цена={price_now:.2f}"
                f"  long={prob_long:.3f}  short={prob_short:.3f}"
                f"  {pos_info}{signal_str}{lvl_ctx}"
            )

            if has_open_position(mt5lib, args.symbol):
                if signal_long or signal_short:
                    log.info("Позиция уже открыта — пропускаем")
            elif trades_today >= args.max_trades_day:
                if signal_long or signal_short:
                    log.info(f"Лимит сделок в день ({args.max_trades_day}) достигнут")
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
                    trades_today   += 1
                    last_trade_time = now
                    log.info(f"Сделок сегодня: {trades_today}/{args.max_trades_day}")

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
