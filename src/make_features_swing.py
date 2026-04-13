"""
Строит датасет для свинг-бота.

Метка: направление цены через N баров.
  1 = цена выросла (close[i+horizon] > close[i])
  0 = цена упала  (close[i+horizon] < close[i])

TP/SL не фиксируются здесь — ставишь любые при запуске бота.

Запуск:
    python src/make_features_swing.py --direction long
    python src/make_features_swing.py --direction short
    python src/make_features_swing.py --direction long --horizon 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from swing_features import build_all_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int, default=20,
                   help="Баров вперёд (default: 20 мин)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


def add_label(df: pd.DataFrame, horizon: int, direction: str) -> pd.DataFrame:
    future_close = df["close"].shift(-horizon)
    current_close = df["close"]

    if direction == "long":
        # 1 = цена через N баров выше текущей
        df["label"] = (future_close > current_close).astype(np.int8)
    else:
        # 1 = цена через N баров ниже текущей
        df["label"] = (future_close < current_close).astype(np.int8)

    # Последние horizon баров не имеют будущего — убираем
    df.loc[df.index[-horizon:], "label"] = np.nan
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(np.int8)
    return df


def main() -> None:
    args = parse_args()
    src = DATA_DIR / f"XAUUSD_TickData_{args.tf}.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"Нет файла: {src}")

    print(f"Читаем {src} ...")
    df = pd.read_parquet(src)
    df = df.dropna(subset=["open", "close", "high", "low"])
    print(f"Строк: {len(df):,}  диапазон: {df.index[0]} → {df.index[-1]}")

    print("Строим свинг-признаки...")
    df = build_all_features(df)

    print(f"Строим метку (direction={args.direction}, horizon={args.horizon} баров) ...")
    df = add_label(df, args.horizon, args.direction)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    out = DATA_DIR / f"dataset_swing_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    df.to_parquet(out)

    pos   = df["label"].sum()
    total = len(df)
    days  = (df.index[-1] - df.index[0]).days

    print(f"\nГотово. Строк: {total:,}")
    print(f"Метка 1 (цена идёт {args.direction}): {pos:,}  ({100*pos/total:.1f}%)")
    print(f"Метка 0 (против):                     {total-pos:,}  ({100*(total-pos)/total:.1f}%)")
    if days > 0:
        print(f"\nПериод: {days} дней (~{days*5//7} торговых)")
    print(f"Признаков: {len(feature_cols)}")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
