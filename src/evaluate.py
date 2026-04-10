"""
Анализ порогов и простой бэктест модели.

Запуск:
    python src/evaluate.py
    python src/evaluate.py --tf M1 --horizon 5 --tp 0.5 --sl 0.5
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS_EXCLUDE = {"open", "high", "low", "close", "volume", "label"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="M1")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.5)
    p.add_argument("--sl", type=float, default=0.5)
    return p.parse_args()


def backtest(df_test: pd.DataFrame, y_prob: np.ndarray, threshold: float,
             tp: float, sl: float) -> dict:
    """
    Простой бэктест:
      - Входим в лонг по close когда prob >= threshold
      - Выход: +tp (прибыль) или -sl (убыток) в $ на унцию золота
      - Не учитывает спред/комиссию (добавим позже)
    """
    signals = y_prob >= threshold
    n_signals = signals.sum()
    if n_signals == 0:
        return {"n_signals": 0, "winrate": 0, "total_pnl": 0, "avg_pnl": 0}

    labels = df_test["label"].values
    # Если метка=1 → сделка дошла до TP, иначе → SL
    pnl_per_trade = np.where(labels[signals] == 1, tp, -sl)

    total = pnl_per_trade.sum()
    winrate = (pnl_per_trade > 0).mean()

    return {
        "threshold": threshold,
        "n_signals": int(n_signals),
        "signal_rate": f"{100 * n_signals / len(signals):.1f}%",
        "winrate": f"{100 * winrate:.1f}%",
        "total_pnl": f"${total:,.0f}",
        "avg_pnl": f"${pnl_per_trade.mean():.3f} / сделку",
        "profit_factor": (
            f"{pnl_per_trade[pnl_per_trade>0].sum() / max(abs(pnl_per_trade[pnl_per_trade<0].sum()), 1e-9):.2f}"
        ),
    }


def main() -> None:
    args = parse_args()
    tag = f"{args.tf}_h{args.horizon}_tp{args.tp}_sl{args.sl}"

    model_path = MODELS_DIR / f"lgbm_{tag}.joblib"
    meta_path = MODELS_DIR / f"meta_{tag}.json"
    data_path = DATA_DIR / f"dataset_{tag}.parquet"

    if not model_path.is_file():
        raise FileNotFoundError(f"Нет модели: {model_path}\nСначала: python src/train.py")

    print("Загружаем модель и данные...")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.read_parquet(data_path)
    feature_cols = meta["feature_cols"]

    cutoff = pd.Timestamp(meta["test_cutoff"])
    df_test = df[df.index > cutoff].copy()

    X_test = df_test[feature_cols]
    y_test = df_test["label"]
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nТестовая выборка: {df_test.index.min()} → {df_test.index.max()}")
    print(f"Строк: {len(df_test):,}   ROC-AUC: {meta['roc_auc']}")

    # ── Анализ порогов
    print("\n" + "=" * 65)
    print(f"{'Порог':>6} | {'Сигналов':>9} | {'% баров':>7} | {'Winrate':>8} | {'Ср. PnL/сделку':>14} | {'Profit Factor':>13}")
    print("-" * 65)

    thresholds = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
    results = []
    for t in thresholds:
        r = backtest(df_test, y_prob, t, args.tp, args.sl)
        if r["n_signals"] == 0:
            continue
        results.append(r)
        print(f"  {t:.2f} | {r['n_signals']:>9,} | {r['signal_rate']:>7} | {r['winrate']:>8} | {r['avg_pnl']:>14} | {r['profit_factor']:>13}")

    print("=" * 65)

    # ── Кривая накопленного PnL при пороге 0.62
    best_thresh = 0.62
    signals = y_prob >= best_thresh
    labels_arr = df_test["label"].values
    pnl_arr = np.where(labels_arr[signals] == 1, args.tp, -args.sl)
    cum_pnl = np.cumsum(pnl_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL кривая
    axes[0].plot(cum_pnl, color="steelblue", linewidth=0.8)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Накопленный PnL (порог={best_thresh}, {len(cum_pnl):,} сделок)")
    axes[0].set_xlabel("Номер сделки")
    axes[0].set_ylabel("PnL, $")
    axes[0].grid(True, alpha=0.3)

    # ROC кривая
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="darkorange", lw=1.5, label=f"ROC-AUC = {meta['roc_auc']}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC кривая")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = PLOTS_DIR / f"eval_{tag}.png"
    plt.savefig(out_plot, dpi=120)
    print(f"\nГрафики сохранены: {out_plot}")

    # ── Вывод рекомендации
    print("\n── Интерпретация ──────────────────────────────────────")
    print(f"  ROC-AUC {meta['roc_auc']} — модель лучше случайного.")
    print("  Порог 0.50 → почти всегда 'покупай' (много сигналов, низкая точность).")
    print("  Порог 0.65–0.70 → меньше сигналов, выше точность — подходит для торговли.")
    print("  Следующий шаг: добавить спред/комиссию (~$0.20–0.30) и проверить реальный PnL.")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
