"""
Признаки для свинг-бота — поиск начала сильных движений (400+ пунктов = $4+).

Фокус: детектировать момент когда рынок готов к сильному импульсу.

Ключевые группы признаков:
  1. Базовые (EMA, ATR, RSI, MACD, BB, объём, время)
  2. Импульс текущей свечи (размер тела, направление)
  3. Расширение волатильности (ATR растёт?)
  4. Пробой уровней (EMA21, EMA50, EMA200)
  5. Накопление-сжатие (squeeze) и его пробой
  6. Сессионный контекст (Лондон, Нью-Йорк — самые сильные движения)
  7. Структура последних N баров (импульс vs. флет)

MIN_BARS_SWING = 250  (нужно для EMA200 + rolling windows)
"""

import numpy as np
import pandas as pd

MIN_BARS_SWING = 250


# ──────────────────────────────────────────────────────────────
# Базовые признаки
# ──────────────────────────────────────────────────────────────

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    vol = df["volume"]

    for w in [5, 10, 20, 50, 200]:
        df[f"ema_{w}"]      = c.ewm(span=w, adjust=False).mean()
        df[f"dist_ema_{w}"] = (c - df[f"ema_{w}"]) / (df[f"ema_{w}"] + 1e-9)

    for w in [5, 20, 50]:
        df[f"sma_{w}"]      = c.rolling(w).mean()
        df[f"dist_sma_{w}"] = (c - df[f"sma_{w}"]) / (df[f"sma_{w}"] + 1e-9)

    prev_c = c.shift(1)
    tr = pd.concat([h - lo,
                    (h - prev_c).abs(),
                    (lo - prev_c).abs()], axis=1).max(axis=1)
    df["atr_7"]   = tr.rolling(7).mean()
    df["atr_14"]  = tr.rolling(14).mean()
    df["atr_20"]  = tr.rolling(20).mean()
    df["atr_50"]  = tr.rolling(50).mean()

    # Расширение волатильности: текущий ATR vs. средний ATR
    df["atr_ratio_7_20"]  = df["atr_7"]  / (df["atr_20"] + 1e-9)
    df["atr_ratio_20_50"] = df["atr_20"] / (df["atr_50"] + 1e-9)

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
    df["bb_pos"]   = (c - bb_mid) / (2 * bb_std + 1e-9)
    df["bb_width"] = (4 * bb_std) / (bb_mid + 1e-9)   # ширина канала

    for lag in [1, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    df["vol_norm"]   = vol / (vol.rolling(20).mean() + 1e-9)
    df["vol_norm_5"] = vol / (vol.rolling(5).mean()  + 1e-9)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


# ──────────────────────────────────────────────────────────────
# Свинг-специфические признаки
# ──────────────────────────────────────────────────────────────

def build_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"]
    o   = df["open"]
    h   = df["high"]
    lo  = df["low"]
    atr = df["atr_20"]

    rng      = h - lo
    body     = c - o
    abs_body = body.abs()
    upper_w  = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_w  = pd.concat([o, c], axis=1).min(axis=1) - lo

    # ── Сила текущей свечи
    df["bar_body_ratio"]  = abs_body / (rng + 1e-9)      # тело / диапазон (0–1)
    df["bar_size_atr"]    = rng / (atr + 1e-9)           # размер в ATR
    df["bar_bullish"]     = (body > 0).astype(float)
    df["bar_bearish"]     = (body < 0).astype(float)

    # Сильный бычий/медвежий бар: тело > 60% диапазона и диапазон > ATR
    df["strong_bull_bar"] = ((abs_body > rng * 0.6) & (rng > atr) & (body > 0)).astype(float)
    df["strong_bear_bar"] = ((abs_body > rng * 0.6) & (rng > atr) & (body < 0)).astype(float)

    # Очень сильный бар (≥ 2× ATR)
    df["impulse_bull"] = ((rng >= atr * 2) & (abs_body > rng * 0.5) & (body > 0)).astype(float)
    df["impulse_bear"] = ((rng >= atr * 2) & (abs_body > rng * 0.5) & (body < 0)).astype(float)

    # ── Структура последних N баров
    for w in [3, 5, 10, 20]:
        # Накопленный возврат
        df[f"sum_ret_{w}"]    = df["ret_1"].rolling(w).sum() if "ret_1" in df.columns else \
                                 np.log(c / c.shift(w))
        # Количество бычьих баров из последних w
        df[f"bull_bars_{w}"]  = (body.shift(1) > 0).rolling(w).sum() / w
        # Средний диапазон / ATR (расширяется ли волатильность)
        df[f"avg_rng_{w}"]    = rng.shift(1).rolling(w).mean() / (atr + 1e-9)

    # ── Пробой EMA уровней
    ema21  = df["ema_21"]  if "ema_21"  in df.columns else c.ewm(span=21,  adjust=False).mean()
    ema50  = df["ema_50"]  if "ema_50"  in df.columns else c.ewm(span=50,  adjust=False).mean()
    ema200 = df["ema_200"] if "ema_200" in df.columns else c.ewm(span=200, adjust=False).mean()

    prev_c = c.shift(1)
    df["cross_ema21_up"]  = ((c > ema21) & (prev_c <= ema21)).astype(float)
    df["cross_ema21_dn"]  = ((c < ema21) & (prev_c >= ema21)).astype(float)
    df["cross_ema50_up"]  = ((c > ema50) & (prev_c <= ema50)).astype(float)
    df["cross_ema50_dn"]  = ((c < ema50) & (prev_c >= ema50)).astype(float)

    # Позиция цены относительно EMA200 (долгосрочный тренд)
    df["above_ema200"]    = (c > ema200).astype(float)
    df["dist_ema200"]     = (c - ema200) / (atr + 1e-9)

    # EMA21 vs EMA50 (среднесрочный тренд)
    df["ema21_above_50"]  = (ema21 > ema50).astype(float)
    df["ema_momentum"]    = (ema21 - ema50) / (atr + 1e-9)

    # ── Сжатие (squeeze) и пробой
    avg_rng_10 = rng.shift(1).rolling(10).mean()
    squeezed   = avg_rng_10 < atr * 0.6
    df["squeezed_10"]     = squeezed.astype(float)
    df["squeeze_bull"]    = (squeezed.shift(1) & (body > 0) & (rng > atr * 1.5)).astype(float)
    df["squeeze_bear"]    = (squeezed.shift(1) & (body < 0) & (rng > atr * 1.5)).astype(float)

    # ── Уровни: High/Low последних N баров (пробой диапазона)
    for w in [10, 20, 50]:
        hi_w = h.shift(1).rolling(w).max()
        lo_w = lo.shift(1).rolling(w).min()
        df[f"break_high_{w}"] = (c > hi_w).astype(float)
        df[f"break_low_{w}"]  = (c < lo_w).astype(float)
        df[f"dist_high_{w}"]  = (c - hi_w) / (atr + 1e-9)
        df[f"dist_low_{w}"]   = (c - lo_w) / (atr + 1e-9)

    # ── Сессионный контекст (UTC)
    hour = df.index.hour
    # Лондон открытие: 07:00–09:00 UTC
    df["london_open"]  = ((hour >= 7)  & (hour < 9)).astype(float)
    # Нью-Йорк открытие: 13:00–15:00 UTC
    df["ny_open"]      = ((hour >= 13) & (hour < 15)).astype(float)
    # Активная торговля: 08:00–17:00 UTC (пересечение Лондон+NY)
    df["active_sess"]  = ((hour >= 8)  & (hour < 17)).astype(float)
    # Азия: 00:00–07:00 UTC (слабые движения для золота)
    df["asia_sess"]    = ((hour >= 0)  & (hour < 7)).astype(float)

    # ── RSI экстремумы (перепродан/перекуплен перед разворотом)
    rsi = df["rsi_14"] if "rsi_14" in df.columns else df.get("rsi_7", pd.Series(50, index=df.index))
    df["rsi_oversold"]   = (rsi < 30).astype(float)
    df["rsi_overbought"] = (rsi > 70).astype(float)
    df["rsi_mid_bull"]   = ((rsi > 50) & (rsi < 70)).astype(float)  # бычий моментум
    df["rsi_mid_bear"]   = ((rsi < 50) & (rsi > 30)).astype(float)  # медвежий моментум

    # ── Объём подтверждение
    df["vol_surge"]   = (df["vol_norm"] > 1.5).astype(float)    # объём > 1.5× среднего
    df["vol_surge_2"] = (df["vol_norm"] > 2.0).astype(float)    # объём > 2× среднего

    # ── Итоговые составные сигналы (признаки-кандидаты на старт свинга)
    # Бычий свинг-кандидат: сильный бар + пробой EMA50 + активная сессия
    df["swing_candidate_bull"] = (
        df["strong_bull_bar"].astype(bool) &
        (df["dist_ema200"] > -5) &   # не слишком далеко ниже EMA200
        df["active_sess"].astype(bool)
    ).astype(float)

    # Медвежий свинг-кандидат
    df["swing_candidate_bear"] = (
        df["strong_bear_bar"].astype(bool) &
        (df["dist_ema200"] < 5) &
        df["active_sess"].astype(bool)
    ).astype(float)

    return df


# ──────────────────────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = build_base_features(df)
    df = build_swing_features(df)
    return df
