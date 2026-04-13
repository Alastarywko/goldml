"""
Обучение LightGBM модели на уровневых признаках.
Модели сохраняются с префиксом "level_".

Запуск:
    python src/train_levels.py
    python src/train_levels.py --direction short
    python src/train_levels.py --horizon 10 --tp 1.0 --sl 0.5

Результат:
    models/level_lgbm_M1_h5_tp0.5_sl0.5_long.joblib
    models/level_meta_M1_h5_tp0.5_sl0.5_long.json
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

# Добавляем scripts/ в путь чтобы импортировать level_features
sys.path.insert(0, str(ROOT / "scripts"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf",        default="M1")
    p.add_argument("--horizon",   type=int,   default=3,
                   help="Должно совпадать с make_features_levels.py (default: 3)")
    p.add_argument("--tp",        type=float, default=0.3,
                   help="Должно совпадать с make_features_levels.py (default: 0.3)")
    p.add_argument("--sl",        type=float, default=0.3,
                   help="Должно совпадать с make_features_levels.py (default: 0.3)")
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Порог для анализа сигналов (не влияет на обучение)")
    return p.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    fname = (f"dataset_levels_{args.tf}_h{args.horizon}"
             f"_tp{args.tp}_sl{args.sl}_{args.direction}.parquet")
    path = DATA_DIR / fname
    if not path.is_file():
        print(f"[!] Файл не найден: {path}")
        print("    Сначала запустите: python src/make_features_levels.py "
              f"--direction {args.direction}")
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
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: lgb.LGBMClassifier, X_test: pd.DataFrame,
             y_test: pd.Series, threshold: float) -> None:
    prob  = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, prob)
    pred  = (prob >= 0.5).astype(int)

    print("=" * 50)
    print(f"Accuracy : {(pred == y_test).mean():.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("=" * 50)
    print(classification_report(y_test, pred,
                                target_names=["нет сигнала", "сигнал"]))

    print(f"\n{'Порог':>6} | {'Сигналов':>9} | {'% баров':>8} | "
          f"{'Winrate':>8} | {'Ср. PnL/сделку':>14}")
    print("-" * 60)
    for thr in [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70]:
        mask = prob >= thr
        n    = mask.sum()
        if n == 0:
            continue
        wr   = y_test[mask].mean() * 100
        pnl  = (y_test[mask].values * 0.5 - (1 - y_test[mask].values) * 0.5).mean()
        print(f"{thr:>6.2f} | {n:>9,} | {100*n/len(y_test):>8.1f}% | "
              f"{wr:>8.1f}% | ${pnl:>+.3f} / сделку")
    print()

    return auc, prob


def feature_importance(model: lgb.LGBMClassifier, feature_cols: list[str]) -> None:
    imp = sorted(zip(feature_cols, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("Топ-20 важных признаков:")
    for name, val in imp[:20]:
        tag = " ← уровень" if any(
            x in name for x in ["day", "session", "near", "broke",
                                 "fakeout", "bounce", "breakout"]
        ) else ""
        print(f"  {name:<30} {val}{tag}")


def save_model(model: lgb.LGBMClassifier, auc: float,
               feature_cols: list[str], args: argparse.Namespace) -> None:
    tag   = f"M1_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}"
    mpath = MODELS_DIR / f"level_lgbm_{tag}.joblib"
    jpath = MODELS_DIR / f"level_meta_{tag}.json"

    joblib.dump(model, mpath)
    meta = {
        "tag":          f"level_{tag}",
        "direction":    args.direction,
        "horizon":      args.horizon,
        "tp":           args.tp,
        "sl":           args.sl,
        "roc_auc":      round(auc, 4),
        "feature_cols": feature_cols,
        "model_type":   "levels",
    }
    with open(jpath, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nМодель сохранена : {mpath}")
    print(f"Метаданные       : {jpath}")


def main() -> None:
    args = parse_args()
    df   = load_dataset(args)

    train_df, test_df = time_split(df)
    feature_cols = get_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    print(f"\nОбучаем LightGBM (уровневые признаки, direction={args.direction}) ...")
    model = train_model(X_train, y_train)

    auc, prob = evaluate(model, X_test, y_test, args.threshold)
    feature_importance(model, feature_cols)
    save_model(model, auc, feature_cols, args)


if __name__ == "__main__":
    main()
