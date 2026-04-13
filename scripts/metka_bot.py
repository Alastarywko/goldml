"""
Торговый бот XAUUSD на основе сигналов индикатора Metka.

Два режима работы:
  1. ЧИСТАЯ МЕТКА (по умолчанию)
     Торгует на каждый сигнал Метки: пин-бар, разворот импульса,
     squeeze breakout, RSI-дивергенция + swing-фильтр.
     Модель не нужна.

  2. МЕТКА + ML-ФИЛЬТР (--use-model)
     Дополнительно проверяет ML-модель: открывает только те сигналы
     Метки, которые модель считает потенциально прибыльными.
     Модели должны быть обучены: python src/train_metka.py --direction long/short

Запуск:
    python scripts/metka_bot.py --dry-run
    python scripts/metka_bot.py --lot 0.1 --tp-points 50 --sl-points 50

    # С ML-фильтром:
    python scripts/metka_bot.py --use-model --dry-run
    python scripts/metka_bot.py --use-model --threshold 0.65

Параметры:
    --symbol         Символ (default: XAUUSD)
    --use-model      Включить ML-фильтр (default: выключен)
    --threshold      Порог вероятности ML-фильтра (default: 0.62)
    --lot            Фиксированный лот (если задан, --risk игнорируется)
    --risk           Риск на сделку в %% от баланса (default: 1.0)
    --tp-points      Тейк-профит в пунктах (default: 50)
    --sl-points      Стоп-лосс в пунктах (default: 50)
    --max-trades-day Макс. сделок в день (default: 50)
    --cooldown       Секунд паузы после сделки (default: 60)
    --dry-run        Не открывать реальные ордера
    --models-dir     Путь к папке models/ (default: ../models)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metka_features import build_all_features, MIN_BARS_METKA  # noqa: E402

BARS_TO_FETCH = max(MIN_BARS_METKA, 150)   # EMA50 + div lookback + запас

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"metka_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"),
    ],
)
log = logging.getLogger("metka_bot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD Metka Bot")
    p.add_argument("--symbol",          default="XAUUSD")
    p.add_argument("--use-model",       action="store_true",
                   help="Включить ML-фильтр поверх сигналов Метки")
    p.add_argument("--threshold",       type=float, default=0.62,
                   help="Порог ML-фильтра (только с --use-model, default: 0.62)")
    p.add_argument("--lot",             type=float, default=None,
                   help="Фиксированный лот (если задан, --risk игнорируется)")
    p.add_argument("--risk",            type=float, default=1.0,
                   help="Риск на сделку, %% от баланса (default: 1.0)")
    p.add_argument("--tp-points",       type=int,   default=50)
    p.add_argument("--sl-points",       type=int,   default=50)
    p.add_argument("--max-trades-day",  type=int,   default=50)
    p.add_argument("--cooldown",        type=int,   default=60,
                   help="Секунд паузы после сделки (default: 60)")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--models-dir",      default=None)
    return p.parse_args()


# ── Загрузка ML-модели (только если --use-model)

def load_metka_model(models_dir: Path, tag: str):
    import joblib
    model_path = models_dir / f"metka_lgbm_{tag}.joblib"
    meta_path  = models_dir / f"metka_meta_{tag}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ── Данные из MT5

def get_bars(mt5lib, symbol: str, n: int) -> pd.DataFrame:
    rates = mt5lib.copy_rates_from_pos(symbol, mt5lib.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Нет данных MT5 для {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
    return df


# ── Лот по риску

def calc_lot(mt5lib, symbol: str, sl_dollar: float, risk_pct: float) -> float:
    info     = mt5lib.account_info()
    sym_info = mt5lib.symbol_info(symbol)
    point    = sym_info.point
    lot_size = sym_info.trade_contract_size
    sl_pts   = sl_dollar / point
    pip_val  = point * lot_size
    lot      = (info.balance * risk_pct / 100) / (sl_pts * pip_val)
    step     = sym_info.volume_step
    lot      = round(lot / step) * step
    return max(sym_info.volume_min, min(sym_info.volume_max, lot))


# ── Ордера

def open_buy(mt5lib, symbol: str, lot: float, tp: float, sl: float,
             dry_run: bool) -> bool:
    ask      = mt5lib.symbol_info_tick(symbol).ask
    sl_price = round(ask - sl, 2)
    tp_price = round(ask + tp, 2)
    if dry_run:
        log.info(f"[DRY-RUN] BUY {lot:.2f} @ {ask:.2f}  SL={sl_price}  TP={tp_price}")
        return True
    result = mt5lib.order_send({
        "action":       mt5lib.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5lib.ORDER_TYPE_BUY,
        "price":        ask,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        20260030,
        "comment":      "metka_bot",
        "type_time":    mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    })
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"BUY {lot:.2f} @ {ask:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
    return False


def open_sell(mt5lib, symbol: str, lot: float, tp: float, sl: float,
              dry_run: bool) -> bool:
    bid      = mt5lib.symbol_info_tick(symbol).bid
    sl_price = round(bid + sl, 2)
    tp_price = round(bid - tp, 2)
    if dry_run:
        log.info(f"[DRY-RUN] SELL {lot:.2f} @ {bid:.2f}  SL={sl_price}  TP={tp_price}")
        return True
    result = mt5lib.order_send({
        "action":       mt5lib.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5lib.ORDER_TYPE_SELL,
        "price":        bid,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        20260031,
        "comment":      "metka_bot_short",
        "type_time":    mt5lib.ORDER_TIME_GTC,
        "type_filling": mt5lib.ORDER_FILLING_IOC,
    })
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"SELL {lot:.2f} @ {bid:.2f}  SL={sl_price}  TP={tp_price}  ticket={result.order}")
        return True
    log.error(f"Ошибка ордера: {result.retcode}  {result.comment}")
    return False


def has_open_position(mt5lib, symbol: str) -> bool:
    pos = mt5lib.positions_get(symbol=symbol)
    return pos is not None and len(pos) > 0


# ── Лог-строка: какие паттерны сработали

def metka_context(row: pd.Series, prob_long: float = None, prob_short: float = None) -> str:
    patterns = []
    for col in ["pin_bar_buy", "pin_bar_sell",
                "momentum_rev_buy", "momentum_rev_sell",
                "squeeze_break_buy", "squeeze_break_sell",
                "rsi_div_buy", "rsi_div_sell"]:
        if col in row.index and row[col]:
            patterns.append(col)
    if "is_strong" in row.index and row["is_strong"]:
        patterns.append("STRONG")
    if "squeezed" in row.index and row["squeezed"]:
        patterns.append("squeezed")
    strength_b = int(row.get("metka_buy_strength", 0))
    strength_s = int(row.get("metka_sell_strength", 0))
    if strength_b or strength_s:
        patterns.append(f"str={strength_b}/{strength_s}")
    parts = ["[" + " | ".join(patterns) + "]"] if patterns else []
    if prob_long is not None:
        parts.append(f"ML: long={prob_long:.3f} short={prob_short:.3f}")
    return "  " + "  ".join(parts) if parts else ""


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

    # ── Режим работы
    use_model    = args.use_model
    model_long   = None
    model_short  = None
    feature_cols = None

    if use_model:
        models_dir = (
            Path(args.models_dir) if args.models_dir
            else Path(__file__).resolve().parent.parent / "models"
        )
        tag_long  = "M1_h5_long"
        tag_short = "M1_h5_short"
        try:
            model_long, meta_long = load_metka_model(models_dir, tag_long)
            feature_cols = meta_long["feature_cols"]
            log.info(f"ML-фильтр LONG : metka_{tag_long}  AUC={meta_long['roc_auc']}")
        except FileNotFoundError as e:
            log.error(f"Не удалось загрузить модель: {e}")
            log.error("Обучите модель: python src/train_metka.py --direction long")
            sys.exit(1)
        try:
            model_short, meta_short = load_metka_model(models_dir, tag_short)
            log.info(f"ML-фильтр SHORT: metka_{tag_short}  AUC={meta_short['roc_auc']}")
        except FileNotFoundError:
            log.warning("Модель SHORT не найдена — торгуем только BUY")

    mode_str = f"Метка + ML-фильтр (порог={args.threshold})" if use_model else "Чистая Метка (без ML)"
    log.info(f"Режим сигналов : {mode_str}")
    log.info(f"Режим торговли : {'DRY-RUN (без ордеров)' if args.dry_run else '⚠️  LIVE'}")
    if args.lot is not None:
        log.info(f"Лот: {args.lot} (фиксированный)")
    else:
        log.info(f"Риск: {args.risk}% от баланса")
    log.info(f"Макс. сделок/день: {args.max_trades_day}  Cooldown: {args.cooldown} сек")
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
                time.sleep(60 - now.second + 2)
                continue

            df = get_bars(mt5lib, args.symbol, n=BARS_TO_FETCH)
            df = build_all_features(df)

            # Берём последнюю строку — проверяем что нет NaN в ключевых полях
            last = df.dropna(subset=["metka_buy", "metka_sell", "atr_20"]).iloc[[-1]]
            if last.empty:
                log.warning("Не хватает данных для расчёта признаков")
                time.sleep(10)
                continue

            last_row = last.iloc[0]

            # ── Сигнал от Метки (детерминированный)
            raw_buy  = bool(last_row["metka_buy"])
            raw_sell = bool(last_row["metka_sell"])

            # ── ML-фильтр (опционально)
            prob_long  = None
            prob_short = None
            signal_long  = raw_buy
            signal_short = raw_sell

            if use_model and (raw_buy or raw_sell):
                if feature_cols is not None:
                    missing = [c for c in feature_cols if c not in last.columns]
                    if missing:
                        log.warning(f"Отсутствуют признаки: {missing[:5]}...")
                    else:
                        X = last[feature_cols]
                        if raw_buy and model_long:
                            prob_long   = model_long.predict_proba(X)[0, 1]
                            signal_long = prob_long >= args.threshold
                        if raw_sell and model_short:
                            prob_short   = model_short.predict_proba(X)[0, 1]
                            signal_short = prob_short >= args.threshold

            # ── Лог
            bar_time   = last.index[-1].strftime("%Y-%m-%d %H:%M")
            positions  = mt5lib.positions_get(symbol=args.symbol)
            pos_info   = f"позиций: {len(positions)}" if positions else "позиций: 0"
            mk_ctx     = metka_context(last_row, prob_long, prob_short)
            signal_str = ("  >>> BUY <<<" if signal_long
                          else "  >>> SELL <<<" if signal_short
                          else ("  (Metka есть, ML блокирует)" if (raw_buy or raw_sell) and use_model
                                else ""))

            log.info(
                f"[{bar_time}]  цена={last_row['close']:.2f}"
                f"  {pos_info}{signal_str}{mk_ctx}"
            )

            # ── Торговля
            if has_open_position(mt5lib, args.symbol):
                if signal_long or signal_short:
                    log.info("Позиция уже открыта — пропускаем")
            elif trades_today >= args.max_trades_day:
                if signal_long or signal_short:
                    log.info(f"Лимит сделок/день ({args.max_trades_day}) достигнут")
            elif last_trade_time and (now - last_trade_time).seconds < args.cooldown:
                if signal_long or signal_short:
                    log.info(f"Cooldown — ждём {args.cooldown - (now - last_trade_time).seconds} сек")
            else:
                lot = args.lot if args.lot is not None else \
                      calc_lot(mt5lib, args.symbol, exec_sl, args.risk)

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
