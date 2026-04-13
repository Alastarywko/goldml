"""
Инверсный свинг-бот — для экспериментов.
Модель BUY → открываем SELL, и наоборот.

Запуск:
    python scripts/swing_inversebot.py --dry-run
    python scripts/swing_inversebot.py --lot 0.1 --threshold 0.55
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swing_features import build_all_features, MIN_BARS_SWING  # noqa: E402

BARS_TO_FETCH = max(MIN_BARS_SWING, 300)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"swing_inversebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"),
    ],
)
log = logging.getLogger("swing_inversebot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD Swing Inverse Bot")
    p.add_argument("--symbol",          default="XAUUSD")
    p.add_argument("--threshold",       type=float, default=0.55)
    p.add_argument("--lot",             type=float, default=None)
    p.add_argument("--risk",            type=float, default=1.0)
    p.add_argument("--tp-points",       type=int,   default=400)
    p.add_argument("--sl-points",       type=int,   default=200)
    p.add_argument("--max-trades-day",  type=int,   default=10)
    p.add_argument("--cooldown",        type=int,   default=300)
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--models-dir",      default=None)
    return p.parse_args()


def load_swing_model(models_dir: Path, tag: str):
    model_path = models_dir / f"swing_lgbm_{tag}.joblib"
    meta_path  = models_dir / f"swing_meta_{tag}.json"
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


def open_buy(mt5lib, symbol: str, lot: float, tp: float, sl: float, dry_run: bool) -> bool:
    ask = mt5lib.symbol_info_tick(symbol).ask
    if dry_run:
        log.info(f"[DRY-RUN] BUY {lot:.2f} @ {ask:.2f}  SL={round(ask-sl,2)}  TP={round(ask+tp,2)}")
        return True
    result = mt5lib.order_send({
        "action": mt5lib.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot,
        "type": mt5lib.ORDER_TYPE_BUY, "price": ask,
        "sl": round(ask - sl, 2), "tp": round(ask + tp, 2),
        "deviation": 10, "magic": 20260060, "comment": "swing_inv_bot",
        "type_time": mt5lib.ORDER_TIME_GTC, "type_filling": mt5lib.ORDER_FILLING_IOC,
    })
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"BUY {lot:.2f} @ {ask:.2f}  ticket={result.order}")
        return True
    log.error(f"Ошибка: {result.retcode}  {result.comment}")
    return False


def open_sell(mt5lib, symbol: str, lot: float, tp: float, sl: float, dry_run: bool) -> bool:
    bid = mt5lib.symbol_info_tick(symbol).bid
    if dry_run:
        log.info(f"[DRY-RUN] SELL {lot:.2f} @ {bid:.2f}  SL={round(bid+sl,2)}  TP={round(bid-tp,2)}")
        return True
    result = mt5lib.order_send({
        "action": mt5lib.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot,
        "type": mt5lib.ORDER_TYPE_SELL, "price": bid,
        "sl": round(bid + sl, 2), "tp": round(bid - tp, 2),
        "deviation": 10, "magic": 20260061, "comment": "swing_inv_short",
        "type_time": mt5lib.ORDER_TIME_GTC, "type_filling": mt5lib.ORDER_FILLING_IOC,
    })
    if result.retcode == mt5lib.TRADE_RETCODE_DONE:
        log.info(f"SELL {lot:.2f} @ {bid:.2f}  ticket={result.order}")
        return True
    log.error(f"Ошибка: {result.retcode}  {result.comment}")
    return False


def has_open_position(mt5lib, symbol: str) -> bool:
    pos = mt5lib.positions_get(symbol=symbol)
    return pos is not None and len(pos) > 0


def run(args: argparse.Namespace) -> None:
    import MetaTrader5 as mt5lib

    if not mt5lib.initialize():
        log.error(f"MT5 не инициализирован: {mt5lib.last_error()}")
        sys.exit(1)
    log.info(f"MT5: {mt5lib.terminal_info().name}  Аккаунт: {mt5lib.account_info().login}")
    log.info("РЕЖИМ: ИНВЕРСНЫЙ свинг (модель BUY → реально SELL, и наоборот)")

    sym_info = mt5lib.symbol_info(args.symbol)
    point    = sym_info.point
    exec_tp  = round(args.tp_points * point, 2)
    exec_sl  = round(args.sl_points * point, 2)

    models_dir = (
        Path(args.models_dir) if args.models_dir
        else Path(__file__).resolve().parent.parent / "models"
    )
    tag_long  = "M1_h20_long"
    tag_short = "M1_h20_short"

    model_long, meta_long = load_swing_model(models_dir, tag_long)
    feature_cols = meta_long["feature_cols"]
    log.info(f"Модель LONG: swing_{tag_long}  AUC={meta_long['roc_auc']}")

    model_short = None
    try:
        model_short, meta_short = load_swing_model(models_dir, tag_short)
        log.info(f"Модель SHORT: swing_{tag_short}  AUC={meta_short['roc_auc']}")
    except FileNotFoundError:
        log.warning("Модель SHORT не найдена")

    log.info(f"Режим: {'DRY-RUN' if args.dry_run else '⚠️  LIVE'}  Порог: {args.threshold}")
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
            last = df.dropna(subset=feature_cols).iloc[[-1]]

            if last.empty:
                log.warning("Не хватает данных для признаков")
                time.sleep(10)
                continue

            X = last[feature_cols]
            prob_long  = model_long.predict_proba(X)[0, 1]
            prob_short = model_short.predict_proba(X)[0, 1] if model_short else 0.0

            # ИНВЕРСИЯ
            inv_sell = prob_long  >= args.threshold
            inv_buy  = prob_short >= args.threshold

            bar_time   = last.index[-1].strftime("%Y-%m-%d %H:%M")
            positions  = mt5lib.positions_get(symbol=args.symbol)
            pos_info   = f"позиций: {len(positions)}" if positions else "позиций: 0"
            signal_str = ("  >>> INV-SELL <<<" if inv_sell
                          else "  >>> INV-BUY <<<" if inv_buy else "")

            log.info(
                f"[{bar_time}]  цена={last.iloc[0]['close']:.2f}"
                f"  long={prob_long:.3f}  short={prob_short:.3f}"
                f"  {pos_info}{signal_str}"
            )

            if has_open_position(mt5lib, args.symbol):
                pass
            elif trades_today >= args.max_trades_day:
                if inv_sell or inv_buy:
                    log.info(f"Лимит/день ({args.max_trades_day}) достигнут")
            elif last_trade_time and (now - last_trade_time).seconds < args.cooldown:
                if inv_sell or inv_buy:
                    log.info(f"Cooldown — {args.cooldown - (now - last_trade_time).seconds} сек")
            else:
                lot = args.lot if args.lot is not None else \
                      calc_lot(mt5lib, args.symbol, exec_sl, args.risk)
                opened = False
                if inv_sell:
                    log.info("ИНВЕРСИЯ: модель BUY → SELL")
                    opened = open_sell(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)
                elif inv_buy:
                    log.info("ИНВЕРСИЯ: модель SELL → BUY")
                    opened = open_buy(mt5lib, args.symbol, lot, exec_tp, exec_sl, args.dry_run)
                if opened:
                    trades_today   += 1
                    last_trade_time = now

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
