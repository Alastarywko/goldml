"""
Строит датасет для базового бота.

Метка: направление цены через N баров.
  1 = цена выросла (close[i+horizon] > close[i])
  0 = цена упала  (close[i+horizon] < close[i])

TP/SL не фиксируются — ставишь любые при запуске бота.

Запуск:
    python src/make_features.py --direction long
    python src/make_features.py --direction short
    python src/make_features.py --direction long --horizon 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int, default=5,
                   help="Баров вперёд (default: 5 мин)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    vol = df["volume"]

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"]      = c.rolling(w).mean()
        df[f"dist_sma_{w}"] = (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    for w in [9, 21]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - lo,
                    (h - prev_c).abs(),
                    (lo - prev_c).abs()], axis=1).max(axis=1)
    for w in [7, 14]:
        df[f"atr_{w}"] = tr.rolling(w).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    for w in [7, 14]:
        ag = gain.ewm(alpha=1/w, adjust=False).mean()
        al = loss.ewm(alpha=1/w, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pos"] = (c - bb_mid) / (2 * bb_std + 1e-9)

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    df["vol_norm"] = vol / (vol.rolling(20).mean() + 1e-9)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


def add_label(df: pd.DataFrame, horizon: int, direction: str) -> pd.DataFrame:
    future_close  = df["close"].shift(-horizon)
    current_close = df["close"]

    if direction == "long":
        df["label"] = (future_close > current_close).astype(np.int8)
    else:
        df["label"] = (future_close < current_close).astype(np.int8)

    df.loc[df.index[-horizon:], "label"] = np.nan
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(np.int8)
    return df


def main() -> None:
    args = parse_args()
    src  = DATA_DIR / f"XAUUSD_TickData_{args.tf}.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"Нет файла: {src}")

    print(f"Читаем {src} ...")
    df = pd.read_parquet(src)
    df = df.dropna(subset=["open", "close", "high", "low"])
    print(f"Строк: {len(df):,}  диапазон: {df.index[0]} → {df.index[-1]}")

    print("Строим признаки...")
    df = add_features(df)

    print(f"Строим метку (direction={args.direction}, horizon={args.horizon} баров)...")
    df = add_label(df, args.horizon, args.direction)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    out = DATA_DIR / f"dataset_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    df.to_parquet(out)

    pos   = df["label"].sum()
    total = len(df)
    print(f"\nГотово. Строк: {total:,}")
    print(f"Метка 1 ({args.direction}): {pos:,}  ({100*pos/total:.1f}%)")
    print(f"Метка 0 (против):          {total-pos:,}  ({100*(total-pos)/total:.1f}%)")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
