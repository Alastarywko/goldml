"""
Строит датасет для metka-бота.

Обучение ТОЛЬКО на барах где сработал сигнал Метки (metka_buy / metka_sell).
Модель отвечает на вопрос: «дойдёт ли эта Metka до моего TP?»

Метка:
  long:  1 если max(high[i+1..i+horizon]) >= close[i] + min_move
  short: 1 если min(low[i+1..i+horizon])  <= close[i] - min_move

Запуск:
    python src/make_features_metka.py --direction long
    python src/make_features_metka.py --direction short
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
    p.add_argument("--horizon",   type=int,   default=8,
                   help="Баров вперёд (default: 8 мин)")
    p.add_argument("--min-move",  type=float, default=0.80,
                   help="Мин. движение $ для OR-условия (80 пунктов = $0.80)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    return p.parse_args()


def add_label(df: pd.DataFrame, horizon: int, direction: str,
              min_move: float = 0.80) -> pd.DataFrame:
    """
    label = 1 если цена достигла min_move за horizon баров:
      long:  max(high[i+1..i+horizon]) >= close[i] + min_move
      short: min(low[i+1..i+horizon])  <= close[i] - min_move

    Вопрос к модели: «дойдёт ли эта Metka до моего TP?»
    Ожидаемый % label=1: ~30–40% (реальная сложность задачи).
    """
    close_vals = df["close"].values
    high_vals  = df["high"].values
    low_vals   = df["low"].values
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - horizon):
        entry    = close_vals[i]
        future_h = high_vals[i+1 : i+horizon+1]
        future_l = low_vals[i+1  : i+horizon+1]

        if direction == "long":
            labels[i] = 1 if future_h.max() >= entry + min_move else 0
        else:
            labels[i] = 1 if future_l.min() <= entry - min_move else 0

    df["label"] = labels
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

    print("Строим базовые + metka признаки...")
    df = build_all_features(df)

    # ── Метка ──────────────────────────────────────────────────
    min_move = args.min_move
    print(f"Строим метку (direction={args.direction}, horizon={args.horizon}, min_move=${min_move})...")
    df = add_label(df, args.horizon, args.direction, min_move=min_move)

    exclude = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols)

    # ── Фильтр: оставляем ТОЛЬКО бары с сигналом Метки ────────
    sig_col = "metka_buy" if args.direction == "long" else "metka_sell"
    total_before = len(df)
    df = df[df[sig_col] == 1].copy()
    print(f"\nОтфильтровано до Metka-сигналов ({sig_col}=1): "
          f"{total_before:,} → {len(df):,} строк "
          f"(~{len(df) / max((df.index[-1]-df.index[0]).days,1):.0f}/день)")

    if len(df) < 500:
        print("[!] Слишком мало сигналов для обучения. Проверь данные.")
        return

    pos   = df["label"].sum()
    total = len(df)
    days  = max((df.index[-1] - df.index[0]).days, 1)
    print(f"\nМетка 1 (синяя/{args.direction}): {pos:,}  ({100*pos/total:.1f}%)")
    print(f"Метка 0 (чёрная):                {total-pos:,}  ({100*(total-pos)/total:.1f}%)")
    print(f"Период: {df.index[0].date()} → {df.index[-1].date()}  ({days} дней)")

    out = DATA_DIR / f"dataset_metka_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    df.to_parquet(out)
    print(f"Признаков: {len(feature_cols)}")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
