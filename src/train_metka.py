"""
Обучение metka-модели — предсказывает НАПРАВЛЕНИЕ движения за N минут.
TP/SL не фиксированы в метке — ставишь любые при запуске бота.

Запуск:
    python src/train_metka.py --direction long
    python src/train_metka.py --direction short

Результат:
    models/metka_lgbm_M1_h8_long.joblib
    models/metka_meta_M1_h8_long.json

Запуск бота (любые TP/SL):
    python scripts/metka_bot.py --use-model --tp-points 80 --sl-points 40
    python scripts/metka_bot.py --use-model --tp-points 100 --sl-points 50
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int,   default=8)
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--threshold", type=float, default=0.55)
    return p.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    fname = f"dataset_metka_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    path  = DATA_DIR / fname
    if not path.is_file():
        print(f"[!] Файл не найден: {path}")
        print("    Сначала запустите: python src/make_features_metka.py "
              f"--direction {args.direction} --horizon {args.horizon}")
        sys.exit(1)
    print(f"Читаем {path} ...")
    return pd.read_parquet(path)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = df.index.normalize().unique().sort_values()
    split = dates[int(len(dates) * 0.66)]
    train = df[df.index < split]
    test  = df[df.index >= split]
    print(f"Train: {train.index[0]} → {train.index[-1]} ({len(train):,} строк)")
    print(f"Test : {test.index[0]}  → {test.index[-1]}  ({len(test):,} строк)")
    return train, test


EXCLUDE = {"open", "high", "low", "close", "volume", "label"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=150,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: lgb.LGBMClassifier, X_test: pd.DataFrame,
             y_test: pd.Series, args: argparse.Namespace) -> tuple[float, np.ndarray]:
    prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    pred = (prob >= 0.5).astype(int)

    print("=" * 55)
    print(f"Accuracy   : {(pred == y_test).mean():.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print("=" * 55)
    print(classification_report(y_test, pred,
                                target_names=["вниз (0)", "вверх (1)"]))

    days = max((X_test.index[-1] - X_test.index[0]).days, 1)

    print(f"\n{'Порог':>6} | {'Сигналов':>9} | {'~в день':>7} | {'Winrate':>8} |"
          f"  PnL/сделку (TP/SL пункты)")
    print("-" * 78)
    for thr in [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70]:
        mask = prob >= thr
        n    = mask.sum()
        if n == 0:
            continue
        wr      = y_test[mask].mean()
        per_day = n / days
        # PnL для популярных комбинаций TP/SL в пунктах
        pnl_50_50   = wr * 50  - (1 - wr) * 50    # tp=50,  sl=50  (1:1)
        pnl_100_50  = wr * 100 - (1 - wr) * 50    # tp=100, sl=50  (2:1)
        pnl_50_30   = wr * 50  - (1 - wr) * 30    # tp=50,  sl=30  (5:3)
        print(f"{thr:>6.2f} | {n:>9,} | {per_day:>7.1f} | {wr*100:>8.1f}% |"
              f"  50/50: {pnl_50_50:+.1f}п  "
              f"100/50: {pnl_100_50:+.1f}п  "
              f"50/30: {pnl_50_30:+.1f}п")
    print()

    # Статистика по Metka-барам
    for mc in ["metka_buy", "metka_sell"]:
        if mc in X_test.columns:
            mask_m = X_test[mc].astype(bool)
            if mask_m.sum() > 0:
                wr_m = y_test[mask_m].mean() * 100
                print(f"Только {mc}: {mask_m.sum():,} баров  winrate={wr_m:.1f}%")
                for thr in [0.55, 0.60, 0.65]:
                    m = mask_m & (prob >= thr)
                    if m.sum() > 0:
                        print(f"  + модель >= {thr}:  {m.sum():,} баров  "
                              f"winrate={y_test[m].mean()*100:.1f}%")

    return auc, prob


def feature_importance(model: lgb.LGBMClassifier, feature_cols: list[str]) -> None:
    imp = sorted(zip(feature_cols, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("\nТоп-20 важных признаков:")
    for name, val in imp[:20]:
        tag = " ← METKA" if any(x in name for x in [
            "metka", "pin_bar", "momentum_rev", "squeeze", "rsi_div",
            "swing_near", "is_strong", "trend_up", "trend_dn", "ema_dist"
        ]) else ""
        print(f"  {name:<38} {val}{tag}")


def save_model(model: lgb.LGBMClassifier, auc: float,
               feature_cols: list[str], args: argparse.Namespace) -> None:
    tag   = f"M1_h{args.horizon}_{args.direction}"
    mpath = MODELS_DIR / f"metka_lgbm_{tag}.joblib"
    jpath = MODELS_DIR / f"metka_meta_{tag}.json"

    joblib.dump(model, mpath)
    meta = {
        "tag":          f"metka_{tag}",
        "direction":    args.direction,
        "horizon":      args.horizon,
        "label_type":   "direction",
        "roc_auc":      round(auc, 4),
        "feature_cols": feature_cols,
        "model_type":   "metka",
    }
    with open(jpath, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nМодель : {mpath}")
    print(f"Мета   : {jpath}")
    print(f"\nЗапуск бота (примеры):")
    print(f"  python scripts\\metka_bot.py --use-model --tp-points 80 --sl-points 40")
    print(f"  python scripts\\metka_bot.py --use-model --tp-points 100 --sl-points 50")


def main() -> None:
    args = parse_args()
    df   = load_dataset(args)

    train_df, test_df = time_split(df)
    feature_cols = get_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    print(f"\nОбучаем LightGBM (metka, direction={args.direction}, "
          f"horizon={args.horizon} баров)")
    model = train_model(X_train, y_train)
    auc, prob = evaluate(model, X_test, y_test, args)
    feature_importance(model, feature_cols)
    save_model(model, auc, feature_cols, args)


if __name__ == "__main__":
    main()
