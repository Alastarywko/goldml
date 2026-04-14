"""
Инверсный бот Metka — для экспериментов.
Логика: BUY-сигнал Метки → открываем SELL, и наоборот.

Два режима работы:
  1. ЧИСТАЯ МЕТКА (по умолчанию)
     Инвертирует каждый сигнал Метки без ML.

  2. МЕТКА + ML-ФИЛЬТР (--use-model)
     Применяет ML-фильтр, затем инвертирует: торгует против
     тех сигналов, которые модель считает высоковероятными.

Запуск:
    python scripts/metka_inversebot.py --dry-run
    python scripts/metka_inversebot.py --use-model --threshold 0.65 --dry-run
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

BARS_TO_FETCH = max(MIN_BARS_METKA, 150)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"metka_inversebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"),
    ],
)
log = logging.getLogger("metka_inversebot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD Metka Inverse Bot")
    p.add_argument("--symbol",          default="XAUUSD")
    p.add_argument("--use-model",       action="store_true",
                   help="Включить ML-фильтр поверх сигналов Метки")
    p.add_argument("--threshold",       type=float, default=0.62)
    p.add_argument("--lot",             type=float, default=None)
    p.add_argument("--risk",            type=float, default=1.0)
    p.add_argument("--tp-points",       type=int,   default=50)
    p.add_argument("--sl-points",       type=int,   default=50)
    p.add_argument("--max-trades-day",  type=int,   default=50)
    p.add_argument("--cooldown",        type=int,   default=60)
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--models-dir",      default=None)
    return p.parse_args()


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


def get_bars(mt5lib, symbol: str, n: int) -> pd.DataFrame:
    rates = mt5lib.copy_rates_from_pos(symbol, mt5lib.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Нет данных MT5 для {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
    return df


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
        "magic":        20260040,
        "comment":      "metka_inv_bot",
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
        "magic":        20260041,
        "comment":      "metka_inv_short",
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
    parts = ["[" + " | ".join(patterns) + "]"] if patterns else []
    if prob_long is not None or prob_short is not None:
        pl = f"{prob_long:.3f}" if prob_long is not None else "n/a"
        ps = f"{prob_short:.3f}" if prob_short is not None else "n/a"
        parts.append(f"ML: long={pl} short={ps}")
    return "  " + "  ".join(parts) if parts else ""


def run(args: argparse.Namespace) -> None:
    import MetaTrader5 as mt5lib

    if not mt5lib.initialize():
        log.error(f"MT5 не инициализирован: {mt5lib.last_error()}")
        sys.exit(1)
    log.info(f"MT5: {mt5lib.terminal_info().name}  Аккаунт: {mt5lib.account_info().login}")
    log.info("РЕЖИМ: ИНВЕРСНЫЙ (Metka BUY → открываем SELL, и наоборот)")

    sym_info = mt5lib.symbol_info(args.symbol)
    point    = sym_info.point
    exec_tp  = round(args.tp_points * point, 2)
    exec_sl  = round(args.sl_points * point, 2)

    use_model   = args.use_model
    model_long  = None
    model_short = None
    feature_cols = None

    if use_model:
        models_dir = (
            Path(args.models_dir) if args.models_dir
            else Path(__file__).resolve().parent.parent / "models"
        )
        tag_long  = "M1_h8_long"
        tag_short = "M1_h8_short"
        try:
            model_long, meta_long = load_metka_model(models_dir, tag_long)
            feature_cols = meta_long["feature_cols"]
            log.info(f"ML-фильтр LONG : metka_{tag_long}  AUC={meta_long['roc_auc']}")
        except FileNotFoundError as e:
            log.error(f"{e}")
            sys.exit(1)
        try:
            model_short, meta_short = load_metka_model(models_dir, tag_short)
            log.info(f"ML-фильтр SHORT: metka_{tag_short}  AUC={meta_short['roc_auc']}")
        except FileNotFoundError:
            log.warning("Модель SHORT не найдена")

    mode_str = f"Метка + ML-фильтр (порог={args.threshold})" if use_model else "Чистая Метка (без ML)"
    log.info(f"Режим сигналов : {mode_str}")
    log.info(f"Режим торговли : {'DRY-RUN' if args.dry_run else '⚠️  LIVE'}")
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
            last = df.dropna(subset=["metka_buy", "metka_sell", "atr_20"]).iloc[[-1]]

            if last.empty:
                log.warning("Не хватает данных для признаков")
                time.sleep(10)
                continue

            last_row = last.iloc[0]
            raw_buy  = bool(last_row["metka_buy"])
            raw_sell = bool(last_row["metka_sell"])

            prob_long    = None
            prob_short   = None
            signal_long  = raw_buy
            signal_short = raw_sell

            if use_model and (raw_buy or raw_sell) and feature_cols:
                missing = [c for c in feature_cols if c not in last.columns]
                if not missing:
                    X = last[feature_cols]
                    if raw_buy and model_long:
                        prob_long   = model_long.predict_proba(X)[0, 1]
                        signal_long = prob_long >= args.threshold
                    if raw_sell and model_short:
                        prob_short   = model_short.predict_proba(X)[0, 1]
                        signal_short = prob_short >= args.threshold

            # ИНВЕРСИЯ: Metka BUY → SELL, Metka SELL → BUY
            inv_sell = signal_long
            inv_buy  = signal_short

            bar_time  = last.index[-1].strftime("%Y-%m-%d %H:%M")
            positions = mt5lib.positions_get(symbol=args.symbol)
            pos_info  = f"позиций: {len(positions)}" if positions else "позиций: 0"
            mk_ctx    = metka_context(last_row, prob_long, prob_short)

            if inv_sell:
                signal_str = "  >>> INV-SELL <<<"
            elif inv_buy:
                signal_str = "  >>> INV-BUY <<<"
            elif raw_buy and use_model and prob_long is not None:
                signal_str = f"  (INV-SELL заблокирован ML: {prob_long:.3f} < {args.threshold})"
            elif raw_sell and use_model and prob_short is not None:
                signal_str = f"  (INV-BUY заблокирован ML: {prob_short:.3f} < {args.threshold})"
            elif raw_buy and use_model:
                signal_str = "  (INV-SELL: модель не загружена)"
            elif raw_sell and use_model:
                signal_str = "  (INV-BUY: модель не загружена)"
            elif raw_buy:
                signal_str = "  (Metka BUY → будет INV-SELL)"
            elif raw_sell:
                signal_str = "  (Metka SELL → будет INV-BUY)"
            else:
                signal_str = ""

            log.info(
                f"[{bar_time}]  цена={last_row['close']:.2f}"
                f"  {pos_info}{signal_str}{mk_ctx}"
            )

            if has_open_position(mt5lib, args.symbol):
                if inv_sell or inv_buy:
                    log.info("  ↳ не открываем: позиция уже открыта")
            elif trades_today >= args.max_trades_day:
                if inv_sell or inv_buy:
                    log.info(f"  ↳ не открываем: лимит {args.max_trades_day} сделок/день достигнут")
            elif last_trade_time and (now - last_trade_time).seconds < args.cooldown:
                if inv_sell or inv_buy:
                    remaining = args.cooldown - (now - last_trade_time).seconds
                    log.info(f"  ↳ не открываем: cooldown, осталось {remaining} сек")
            else:
                lot = args.lot if args.lot is not None else \
                      calc_lot(mt5lib, args.symbol, exec_sl, args.risk)

                opened = False
                if inv_sell:
                    log.info("ИНВЕРСИЯ: Metka BUY → открываем SELL")
                    opened = open_sell(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)
                elif inv_buy:
                    log.info("ИНВЕРСИЯ: Metka SELL → открываем BUY")
                    opened = open_buy(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)

                if opened:
                    trades_today   += 1
                    last_trade_time = now
                    log.info(f"Сделок сегодня: {trades_today}/{args.max_trades_day}")

            time.sleep(52)

        except KeyboardInterrupt:
            log.info("Остановка (Ctrl+C)")
            break
        except Exception as e:
            log.error(f"Ошибка: {e}", exc_info=True)
            time.sleep(15)

    mt5lib.shutdown()
    log.info("MT5 отключён")


if __name__ == "__main__":
    args = parse_args()
    run(args)
