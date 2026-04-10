"""
Расчёт признаков — используется и при обучении (Ubuntu) и в боте (Windows).
Функция build_features(df) принимает DataFrame с колонками:
  open, high, low, close, volume  и DatetimeIndex
и возвращает DataFrame с признаками (без меток).
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    vol = df["volume"]

    # Скользящие средние
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = c.rolling(w).mean()
        df[f"dist_sma_{w}"] = (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # EMA
    for w in [9, 21]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # ATR
    prev_c = c.shift(1)
    tr = pd.concat([
        h - lo,
        (h - prev_c).abs(),
        (lo - prev_c).abs(),
    ], axis=1).max(axis=1)
    for w in [7, 14]:
        df[f"atr_{w}"] = tr.rolling(w).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for w in [7, 14]:
        avg_gain = gain.ewm(alpha=1 / w, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / w, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pos"] = (c - bb_mid) / (2 * bb_std + 1e-9)

    # Log returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    # Объём
    df["vol_norm"] = vol / (vol.rolling(20).mean() + 1e-9)

    # Время
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


# Минимум баров для корректного расчёта всех признаков
MIN_BARS = 60
