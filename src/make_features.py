"""
Строит признаки и метку из M1 OHLCV и сохраняет dataset.parquet.

Запуск:
    python src/make_features.py                          # лонг (по умолчанию)
    python src/make_features.py --direction short        # шорт
    python src/make_features.py --tf M1 --horizon 5 --tp 0.5 --sl 0.5

Параметры:
    --tf          таймфрейм файла (M1 / M5 / H1), по умолчанию M1
    --horizon     сколько баров смотреть вперёд для расчёта метки (default 5)
    --tp          тейк-профит в $ (default 0.5)
    --sl          стоп-лосс в $ (default 0.5)
    --direction   long или short (default long)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="M1")
    p.add_argument("--horizon", type=int, default=5,
                   help="Горизонт: сколько баров вперёд смотрим")
    p.add_argument("--tp", type=float, default=0.5,
                   help="Тейк-профит ($): на сколько должна вырасти цена за horizon баров")
    p.add_argument("--sl", type=float, default=0.5,
                   help="Стоп-лосс ($): насколько может упасть цена за horizon баров")
    p.add_argument("--direction", default="long", choices=["long", "short"],
                   help="Направление сделки: long (BUY) или short (SELL)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Технические признаки
# ──────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    vol = df["volume"]

    # Скользящие средние
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = c.rolling(w).mean()
        df[f"dist_sma_{w}"] = (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # EMA
    for w in [9, 21]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # ATR (Average True Range) — волатильность
    prev_c = c.shift(1)
    tr = pd.concat([
        h - lo,
        (h - prev_c).abs(),
        (lo - prev_c).abs(),
    ], axis=1).max(axis=1)
    for w in [7, 14]:
        df[f"atr_{w}"] = tr.rolling(w).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for w in [7, 14]:
        avg_gain = gain.ewm(alpha=1 / w, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / w, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (расстояние от средней, нормировано на ширину)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pos"] = (c - bb_mid) / (2 * bb_std + 1e-9)

    # Доходности (log returns)
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    # Объём
    df["vol_norm"] = vol / (vol.rolling(20).mean() + 1e-9)

    # Час дня и день недели (цикличность)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


# ──────────────────────────────────────────────────────────────
# Метка
# ──────────────────────────────────────────────────────────────

def add_label(df: pd.DataFrame, horizon: int, tp: float, sl: float,
              direction: str = "long") -> pd.DataFrame:
    """
    direction='long':
        Метка = 1 если за horizon баров цена выросла на >= tp ДО падения на >= sl

    direction='short':
        Метка = 1 если за horizon баров цена упала на >= tp ДО роста на >= sl
    """
    entry = df["close"].values
    high_f = df["high"].values
    low_f = df["low"].values
    labels = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df) - horizon):
        e = entry[i]
        hit_tp = False
        hit_sl = False
        for j in range(i + 1, i + horizon + 1):
            if direction == "long":
                if high_f[j] - e >= tp:
                    hit_tp = True
                    break
                if e - low_f[j] >= sl:
                    hit_sl = True
                    break
            else:  # short
                if e - low_f[j] >= tp:
                    hit_tp = True
                    break
                if high_f[j] - e >= sl:
                    hit_sl = True
                    break
        if hit_tp and not hit_sl:
            labels[i] = 1

    df["label"] = labels
    return df


def main() -> None:
    args = parse_args()
    src = DATA_DIR / f"XAUUSD_TickData_{args.tf}.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"Нет файла: {src}\nЗапустите сначала: python src/ticks_to_ohlcv.py --tf {args.tf}")

    print(f"Читаем {src} ...")
    df = pd.read_parquet(src)
    print(f"Строк: {len(df):,}  диапазон: {df.index[0]} → {df.index[-1]}")

    # Только бары с нормальными данными (убираем явные пропуски открытых часов)
    df = df.dropna(subset=["open", "close", "high", "low"])

    print("Строим признаки...")
    df = add_features(df)

    print(f"Строим метку (direction={args.direction}, horizon={args.horizon}, tp={args.tp}, sl={args.sl}) ...")
    df = add_label(df, horizon=args.horizon, tp=args.tp, sl=args.sl, direction=args.direction)

    # Убираем строки где признаки ещё NaN (начало рядов скользящих средних)
    feature_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume", "label")]
    df = df.dropna(subset=feature_cols)

    out = DATA_DIR / f"dataset_{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}.parquet"
    df.to_parquet(out)

    pos = df["label"].sum()
    neg = (df["label"] == 0).sum()
    direction_label = "шорт" if args.direction == "short" else "лонг"
    print(f"\nГотово. Строк: {len(df):,}")
    print(f"Метка 1 ({direction_label}): {pos:,}  ({100*pos/len(df):.1f}%)")
    print(f"Метка 0 (нет): {neg:,}  ({100*neg/len(df):.1f}%)")
    print(f"Сохранено: {out}")
    print("\nПризнаки:", feature_cols[:10], "...")


if __name__ == "__main__":
    main()
