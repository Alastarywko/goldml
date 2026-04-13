"""
Обучение свинг-модели — предсказывает НАПРАВЛЕНИЕ движения за N минут.
TP/SL не фиксированы в метке — ставишь любые при запуске бота.

Запуск:
    python src/train_swing.py --direction long
    python src/train_swing.py --direction short
    python src/train_swing.py --direction long --horizon 20

Результат:
    models/swing_lgbm_M1_h20_long.joblib
    models/swing_meta_M1_h20_long.json

При запуске бота TP/SL задаёшь сам:
    python scripts/swing_bot.py --tp-points 400 --sl-points 200 --threshold 0.60
    python scripts/swing_bot.py --tp-points 400 --sl-points 100 --threshold 0.60
    python scripts/swing_bot.py --tp-points 600 --sl-points 200 --threshold 0.60
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
    p.add_argument("--horizon",   type=int,   default=20)
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Порог для оценочной таблицы (default: 0.55)")
    return p.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    fname = f"dataset_swing_{args.tf}_h{args.horizon}_{args.direction}.parquet"
    path  = DATA_DIR / fname
    if not path.is_file():
        print(f"[!] Файл не найден: {path}")
        print("    Сначала запустите: python src/make_features_swing.py "
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
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        # Без class_weight=balanced — классы почти равны (50/50 по направлению)
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

    # Таблица по порогам — показывает winrate при разных TP/SL
    print(f"\n{'Порог':>6} | {'Сигналов':>9} | {'~в день':>7} | {'Winrate':>8} |"
          f"  Ожид. PnL (примеры TP/SL)")
    print("-" * 75)
    for thr in [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70]:
        mask = prob >= thr
        n    = mask.sum()
        if n == 0:
            continue
        wr      = y_test[mask].mean()
        per_day = n / days
        # Показываем PnL для нескольких популярных комбинаций TP/SL
        pnl_400_200 = wr * 4.0 - (1 - wr) * 2.0   # tp=$4, sl=$2
        pnl_400_100 = wr * 4.0 - (1 - wr) * 1.0   # tp=$4, sl=$1
        pnl_200_100 = wr * 2.0 - (1 - wr) * 1.0   # tp=$2, sl=$1
        print(f"{thr:>6.2f} | {n:>9,} | {per_day:>7.1f} | {wr*100:>8.1f}% |"
              f"  400/200: ${pnl_400_200:+.2f}  "
              f"400/100: ${pnl_400_100:+.2f}  "
              f"200/100: ${pnl_200_100:+.2f}")
    print()
    print("  Столбцы PnL = ожидаемый доход на 1 доллар лота за сделку")
    print("  (multiply на размер лота × 100 для реальной прибыли)")

    return auc, prob


def feature_importance(model: lgb.LGBMClassifier, feature_cols: list[str]) -> None:
    imp = sorted(zip(feature_cols, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    swing_tags = {
        "impulse", "strong_bull", "strong_bear", "squeeze",
        "break_high", "break_low", "london", "ny_open", "active_sess",
        "swing_candidate", "atr_ratio", "bull_bars", "avg_rng",
        "cross_ema", "dist_ema200", "ema_momentum", "vol_surge", "bb_width"
    }
    print("\nТоп-20 важных признаков:")
    for name, val in imp[:20]:
        tag = " ← SWING" if any(t in name for t in swing_tags) else ""
        print(f"  {name:<38} {val}{tag}")


def save_model(model: lgb.LGBMClassifier, auc: float,
               feature_cols: list[str], args: argparse.Namespace) -> None:
    tag   = f"M1_h{args.horizon}_{args.direction}"
    mpath = MODELS_DIR / f"swing_lgbm_{tag}.joblib"
    jpath = MODELS_DIR / f"swing_meta_{tag}.json"

    joblib.dump(model, mpath)
    meta = {
        "tag":          f"swing_{tag}",
        "direction":    args.direction,
        "horizon":      args.horizon,
        "label_type":   "direction",
        "roc_auc":      round(auc, 4),
        "feature_cols": feature_cols,
        "model_type":   "swing",
    }
    with open(jpath, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nМодель : {mpath}")
    print(f"Мета   : {jpath}")
    print(f"\nЗапуск бота (примеры):")
    print(f"  python scripts\\swing_bot.py --tp-points 400 --sl-points 200 --threshold 0.60")
    print(f"  python scripts\\swing_bot.py --tp-points 400 --sl-points 100 --threshold 0.60")
    print(f"  python scripts\\swing_bot.py --tp-points 600 --sl-points 200 --threshold 0.65")


def main() -> None:
    args = parse_args()
    df   = load_dataset(args)

    train_df, test_df = time_split(df)
    feature_cols = get_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    pos_rate = y_train.mean() * 100
    print(f"\nОбучаем LightGBM (направление, direction={args.direction}, "
          f"horizon={args.horizon} баров)")
    print(f"Метка 1 (вверх) в train: {y_train.sum():,} ({pos_rate:.1f}%)")
    print(f"Метка 0 (вниз)  в train: {(1-y_train).sum():,} ({100-pos_rate:.1f}%)")

    model = train_model(X_train, y_train)
    auc, prob = evaluate(model, X_test, y_test, args)
    feature_importance(model, feature_cols)
    save_model(model, auc, feature_cols, args)


if __name__ == "__main__":
    main()
