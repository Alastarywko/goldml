"""
Обучает LightGBM на dataset_*.parquet, сохраняет модель и метрики.

Запуск:
    python src/train.py                          # лонг (по умолчанию)
    python src/train.py --direction short        # шорт
    python src/train.py --tf M1 --horizon 5 --tp 0.5 --sl 0.5 --test-years 2
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="M1")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.5)
    p.add_argument("--sl", type=float, default=0.5)
    p.add_argument("--test-years", type=float, default=2.0,
                   help="Сколько последних лет оставить на тест (default 2)")
    p.add_argument("--direction", default="long", choices=["long", "short"],
                   help="Направление: long или short")
    return p.parse_args()


FEATURE_COLS_EXCLUDE = {"open", "high", "low", "close", "volume", "label"}


def main() -> None:
    args = parse_args()

    src = DATA_DIR / f"dataset_{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}.parquet"
    if not src.is_file():
        raise FileNotFoundError(
            f"Нет датасета: {src}\n"
            f"Запустите сначала: python src/make_features.py --direction {args.direction}"
        )

    print(f"Читаем {src} ...")
    df = pd.read_parquet(src)

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = df[feature_cols]
    y = df["label"]

    # ── Разбивка train/test ПО ВРЕМЕНИ (не random, иначе утечка из будущего)
    cutoff = df.index.max() - pd.DateOffset(years=args.test_years)
    train_mask = df.index <= cutoff
    test_mask = df.index > cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nРазбивка по времени:")
    print(f"  Train: {df.index[train_mask].min()} → {df.index[train_mask].max()}  ({train_mask.sum():,} строк)")
    print(f"  Test:  {df.index[test_mask].min()} → {df.index[test_mask].max()}  ({test_mask.sum():,} строк)")

    # ── Обучение LightGBM
    print("\nОбучаем LightGBM ...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # ── Метрики
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print("=" * 50)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["нет сигнала", "лонг"]))

    # ── Важность признаков (топ-15)
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Топ-15 важных признаков:")
    print(fi.head(15).to_string())

    # ── Сохранение
    tag = f"{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}_{args.direction}"
    model_path = MODELS_DIR / f"lgbm_{tag}.joblib"
    meta_path = MODELS_DIR / f"meta_{tag}.json"

    joblib.dump(model, model_path)
    meta = {
        "tf": args.tf,
        "horizon": args.horizon,
        "tp": args.tp,
        "sl": args.sl,
        "direction": args.direction,
        "feature_cols": feature_cols,
        "test_cutoff": str(cutoff.date()),
        "accuracy": round(acc, 4),
        "roc_auc": round(roc, 4),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nМодель сохранена : {model_path}")
    print(f"Метаданные       : {meta_path}")


if __name__ == "__main__":
    main()
