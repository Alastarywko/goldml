"""
Строит датасет для metka-бота.

Метка: направление цены через N баров.
  1 = цена выросла (close[i+horizon] > close[i])
  0 = цена упала  (close[i+horizon] < close[i])

TP/SL не фиксируются — ставишь любые при запуске бота.

Запуск:
    python src/make_features_metka.py --direction long
    python src/make_features_metka.py --direction short
    python src/make_features_metka.py --direction long --horizon 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from metka_features import build_all_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int, default=5,
                   help="Баров вперёд (default: 5 мин)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


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
    src = DATA_DIR / f"XAUUSD_TickData_{args.tf}.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"Нет файла: {src}")

    print(f"Читаем {src} ...")
    df = pd.read_parquet(src)
    df = df.dropna(subset=["open", "close", "high", "low"])
    print(f"Строк: {len(df):,}  диапазон: {df.index[0]} → {df.index[-1]}")

    print("Строим базовые + metka признаки...")
    df = build_all_features(df)

    print(f"Строим метку (direction={args.direction}, horizon={args.horizon} баров)...")
    df = add_label(df, args.horizon, args.direction)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    out = DATA_DIR / f"dataset_metka_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    df.to_parquet(out)

    pos   = df["label"].sum()
    total = len(df)
    print(f"\nГотово. Строк: {total:,}")
    print(f"Метка 1 ({args.direction}): {pos:,}  ({100*pos/total:.1f}%)")
    print(f"Метка 0 (против):          {total-pos:,}  ({100*(total-pos)/total:.1f}%)")

    buy_sigs  = df["metka_buy"].sum()
    sell_sigs = df["metka_sell"].sum()
    days = max((df.index[-1] - df.index[0]).days, 1)
    print(f"\nMetka BUY  сигналов в истории: {int(buy_sigs):,}  (~{buy_sigs/days:.0f}/день)")
    print(f"Metka SELL сигналов в истории: {int(sell_sigs):,}  (~{sell_sigs/days:.0f}/день)")
    print(f"Признаков: {len(feature_cols)}")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
