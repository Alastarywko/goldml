"""
Скальпинговые уровневые признаки для M1.
Используется и при обучении (Ubuntu) и в боте (Windows).

Уровни — короткие скользящие окна (все в M1-барах):
  micro  =  5 баров  ( 5 минут) — самый локальный
  short  = 15 баров  (15 минут) — пробой / отбой внутри четверти часа
  medium = 30 баров  (30 минут) — полчасовой диапазон

Цель: до 10 сигналов в час (один сигнал каждые 6–10 минут).

build_all_features(df) принимает DataFrame с колонками:
  open, high, low, close, volume  и DatetimeIndex
и возвращает DataFrame со всеми признаками.
"""

import numpy as np
import pandas as pd

# Достаточно 100 баров для расчёта всех признаков.
# Бот запрашивает 250 (для SMA200 фильтра тренда).
MIN_BARS_LEVELS = 100


# ──────────────────────────────────────────────────────────────
# Базовые признаки
# ──────────────────────────────────────────────────────────────

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    vol = df["volume"]

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"]      = c.rolling(w).mean()
        df[f"dist_sma_{w}"] = (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    for w in [9, 21]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    for w in [7, 14]:
        df[f"atr_{w}"] = tr.rolling(w).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    for w in [7, 14]:
        ag = gain.ewm(alpha=1/w, adjust=False).mean()
        al = loss.ewm(alpha=1/w, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pos"] = (c - bb_mid) / (2 * bb_std + 1e-9)

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    df["vol_norm"] = vol / (vol.rolling(20).mean() + 1e-9)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


# ──────────────────────────────────────────────────────────────
# Скальпинговые уровневые признаки
# ──────────────────────────────────────────────────────────────

def build_level_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    atr = df["atr_14"]
    vol = df["vol_norm"]

    # Три скальпинговых горизонта
    levels = {
        "micro":  5,   #  5 минут — самый быстрый
        "short":  15,  # 15 минут — основной скальп
        "medium": 30,  # 30 минут — полчасовой диапазон
    }

    for name, w in levels.items():
        rng_high = h.rolling(w, min_periods=w // 2).max()
        rng_low  = lo.rolling(w, min_periods=w // 2).min()

        # Расстояние до уровня нормировано на ATR
        df[f"dist_{name}_high"] = (rng_high - c) / (atr + 1e-9)
        df[f"dist_{name}_low"]  = (c - rng_low)  / (atr + 1e-9)

        # Нахождение у уровня (в пределах 0.3 ATR — тесный для скальпинга)
        df[f"near_{name}_high"] = (df[f"dist_{name}_high"].abs() < 0.3).astype(float)
        df[f"near_{name}_low"]  = (df[f"dist_{name}_low"].abs()  < 0.3).astype(float)

        # Пробой: цена закрылась за предыдущим уровнем + объём хотя бы 1.2×
        prev_high = rng_high.shift(1)
        prev_low  = rng_low.shift(1)
        df[f"breakout_up_{name}"]   = ((c > prev_high) & (vol > 1.2)).astype(float)
        df[f"breakout_down_{name}"] = ((c < prev_low)  & (vol > 1.2)).astype(float)

        # Ложный пробой: 2 бара прокалывали уровень, но текущий бар вернулся + объём слабый
        df[f"fakeout_up_{name}"]   = (
            (h.rolling(2).max() > prev_high) & (c < prev_high) & (vol < 0.8)
        ).astype(float)
        df[f"fakeout_down_{name}"] = (
            (lo.rolling(2).min() < prev_low) & (c > prev_low) & (vol < 0.8)
        ).astype(float)

    # ── Готовые скальпинговые сигналы

    # ОТБОЙ от short (15-мин) уровня — основной скальп-паттерн
    df["bounce_buy"] = (
        df["near_short_low"].astype(bool) &
        (df["rsi_7"] < 38) &
        (vol < 0.9)
    ).astype(float)

    df["bounce_sell"] = (
        df["near_short_high"].astype(bool) &
        (df["rsi_7"] > 62) &
        (vol < 0.9)
    ).astype(float)

    # ПРОБОЙ short уровня с объёмом + подтверждение MACD
    df["breakout_signal_up"] = (
        df["breakout_up_short"].astype(bool) &
        (df["macd_hist"] > 0) &
        (df["atr_7"] > df["atr_14"])      # ATR растёт → импульс
    ).astype(float)

    df["breakout_signal_down"] = (
        df["breakout_down_short"].astype(bool) &
        (df["macd_hist"] < 0) &
        (df["atr_7"] > df["atr_14"])
    ).astype(float)

    # ЛОЖНЫЙ ПРОБОЙ micro (5-мин) уровня — быстрый разворот
    df["fakeout_signal_up"] = (
        df["fakeout_up_micro"].astype(bool) &
        (df["rsi_7"] > 68)
    ).astype(float)

    df["fakeout_signal_down"] = (
        df["fakeout_down_micro"].astype(bool) &
        (df["rsi_7"] < 32)
    ).astype(float)

    # Количество одновременно сработавших уровней (сила зоны)
    near_cols_high = [f"near_{n}_high" for n in levels]
    near_cols_low  = [f"near_{n}_low"  for n in levels]
    df["confluence_high"] = df[near_cols_high].sum(axis=1)  # 0–3
    df["confluence_low"]  = df[near_cols_low].sum(axis=1)   # 0–3

    return df


# ──────────────────────────────────────────────────────────────
# Главная функция — все признаки вместе
# ──────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = build_base_features(df)
    df = build_level_features(df)
    return df
