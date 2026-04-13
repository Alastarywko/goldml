"""
Строит датасет для metka-модели.

Признаки: базовые + паттерны Метки (пин-бар, разворот импульса,
          squeeze breakout, RSI дивергенция, swing-фильтр).
Метка: цена достигнет TP раньше SL за N баров (1 = синий/профит, 0 = чёрный/убыток).

Запуск:
    python src/make_features_metka.py --direction long
    python src/make_features_metka.py --direction short
    python src/make_features_metka.py --direction long --horizon 10 --tp 1.0 --sl 0.5
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
    p.add_argument("--horizon",   type=int,   default=5,
                   help="Баров вперёд (default: 5)")
    p.add_argument("--tp",        type=float, default=0.5,
                   help="Тейк-профит $ (default: 0.5)")
    p.add_argument("--sl",        type=float, default=0.5,
                   help="Стоп-лосс $ (default: 0.5)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


def add_label(df: pd.DataFrame, horizon: int, tp: float, sl: float,
              direction: str) -> pd.DataFrame:
    entry  = df["close"].values
    high_f = df["high"].values
    low_f  = df["low"].values
    labels = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df) - horizon):
        e = entry[i]
        hit_tp = hit_sl = False
        for j in range(i + 1, i + horizon + 1):
            if direction == "long":
                if high_f[j] - e >= tp: hit_tp = True; break
                if e - low_f[j]  >= sl: hit_sl = True; break
            else:
                if e - low_f[j]  >= tp: hit_tp = True; break
                if high_f[j] - e >= sl: hit_sl = True; break
        if hit_tp and not hit_sl:
            labels[i] = 1

    df["label"] = labels
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

    print(f"Строим метку (direction={args.direction}, "
          f"horizon={args.horizon}, tp={args.tp}, sl={args.sl}) ...")
    df = add_label(df, args.horizon, args.tp, args.sl, args.direction)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    out = (DATA_DIR /
           f"dataset_metka_{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}.parquet")
    df.to_parquet(out)

    pos = df["label"].sum()
    neg = (df["label"] == 0).sum()
    print(f"\nГотово. Строк: {len(df):,}")
    print(f"Метка 1 ({args.direction}): {pos:,}  ({100*pos/len(df):.1f}%)")
    print(f"Метка 0 (чёрная):           {neg:,}  ({100*neg/len(df):.1f}%)")
    print(f"Сохранено: {out}")

    # Показываем сколько баров с Metka-сигналом
    buy_sigs  = df["metka_buy"].sum()
    sell_sigs = df["metka_sell"].sum()
    print(f"\nМеток BUY  в истории: {int(buy_sigs):,}")
    print(f"Меток SELL в истории: {int(sell_sigs):,}")
    print(f"Признаков всего: {len(feature_cols)}")
    print("Metka признаки:", [c for c in feature_cols if "metka" in c or "pin" in c
                               or "momentum" in c or "squeeze" in c or "rsi_div" in c
                               or "swing" in c or "trend" in c or "ema_dist" in c])


if __name__ == "__main__":
    main()
