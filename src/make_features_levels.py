"""
Строит скальпинговые уровневые признаки + базовые признаки и метку.
Результат сохраняется в data/dataset_levels_*.parquet

Скальпинговые горизонты уровней (M1-бары):
  micro  =  5 баров ( 5 мин) — быстрый ложный пробой
  short  = 15 баров (15 мин) — основной отбой/пробой
  medium = 30 баров (30 мин) — полчасовой диапазон

Рекомендуемые параметры для скальпинга (до 10 сделок/час):
  --horizon 3   (смотрим 3 бара = 3 минуты вперёд)
  --tp 0.3      (тейк $0.30 — достижим за 3 мин)
  --sl 0.3      (стоп $0.30)

Стандартный запуск:
    python src/make_features_levels.py --direction long  --horizon 3 --tp 0.3 --sl 0.3
    python src/make_features_levels.py --direction short --horizon 3 --tp 0.3 --sl 0.3

Более широкий вариант:
    python src/make_features_levels.py --direction long
    python src/make_features_levels.py --direction short
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SRC_DIR  = ROOT / "src"

# Импортируем базовые признаки
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from level_features import build_all_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int,   default=3,
                   help="Баров вперёд для метки (3 = 3 мин, рекомендуется для скальпинга)")
    p.add_argument("--tp",        type=float, default=0.3,
                   help="Тейк-профит $ (0.3 рекомендуется для скальпинга)")
    p.add_argument("--sl",        type=float, default=0.3,
                   help="Стоп-лосс $ (0.3 рекомендуется для скальпинга)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


def add_label(df: pd.DataFrame, horizon: int, tp: float, sl: float,
              direction: str = "long") -> pd.DataFrame:
    entry  = df["close"].values
    high_f = df["high"].values
    low_f  = df["low"].values
    labels = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df) - horizon):
        e = entry[i]
        hit_tp = hit_sl = False
        for j in range(i + 1, i + horizon + 1):
            if direction == "long":
                if high_f[j] - e >= tp:  hit_tp = True; break
                if e - low_f[j]  >= sl:  hit_sl = True; break
            else:
                if e - low_f[j]  >= tp:  hit_tp = True; break
                if high_f[j] - e >= sl:  hit_sl = True; break
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

    print("Строим базовые + уровневые признаки...")
    df = build_all_features(df)

    print(f"Строим метку (direction={args.direction}, horizon={args.horizon}, "
          f"tp={args.tp}, sl={args.sl}) ...")
    df = add_label(df, args.horizon, args.tp, args.sl, args.direction)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    out = DATA_DIR / f"dataset_levels_{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}.parquet"
    df.to_parquet(out)

    pos = df["label"].sum()
    neg = (df["label"] == 0).sum()
    print(f"\nГотово. Строк: {len(df):,}")
    print(f"Метка 1 ({args.direction}): {pos:,}  ({100*pos/len(df):.1f}%)")
    print(f"Метка 0 (нет):             {neg:,}  ({100*neg/len(df):.1f}%)")
    print(f"Сохранено: {out}")
    print(f"Признаков: {len(feature_cols)}")
    print("Уровневые признаки:", [c for c in feature_cols if any(
        x in c for x in ["day", "session", "near", "broke", "fakeout", "bounce", "breakout"]
    )])


if __name__ == "__main__":
    main()
