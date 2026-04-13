"""
Признаки на основе логики индикатора Metka v28 (дефолтные настройки).

Реализует те же паттерны что Metka рисует на графике:
  - Пин-бар (Method 2)
  - Разворот импульса (Method 3)
  - Squeeze Breakout (Method 4, HTF заменён на M1 EMA21/50)
  - RSI Дивергенция (Method 5, упрощённая)
  - Swing-фильтр (Filter 4, включён по умолчанию)

Дефолтные параметры Metka:
  EMA fast=21, slow=50, ATR=20
  Pin bar: мин. диапазон 2× ATR
  Cooldown: 7 баров (применяется в train через метку)
  Swing: окно 15 баров, допуск 1.5 ATR
  RSI период: 14
  Объём, ADX, BB: ВЫКЛЮЧЕНЫ

build_all_features(df) → DataFrame со всеми признаками.
MIN_BARS_METKA = 120
"""

import numpy as np
import pandas as pd

MIN_BARS_METKA = 120   # EMA50 + div lookback 30 + DIV_SW 3 + запас

# Параметры Metka (дефолт)
_EMA_FAST   = 21
_EMA_SLOW   = 50
_ATR_PERIOD = 20
_SPIKE_ATR  = 2.0       # Pin bar: мин. размер в ATR
_REV_LB     = 5         # Momentum reversal: lookback
_SQZ_LB     = 6         # Squeeze: окно для avg range
_DIV_LB     = 30        # RSI div: окно поиска
_DIV_SW     = 3         # RSI div: ширина свинга
_RSI_PERIOD = 14
_SWING_BARS = 15        # Swing filter: окно
_SWING_ATR  = 1.5       # Swing filter: допуск в ATR


# ──────────────────────────────────────────────────────────────
# Базовые признаки (те же что в других модулях)
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

    for w in [_EMA_FAST, _EMA_SLOW]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    df["atr_7"]  = tr.rolling(7).mean()
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_20"] = tr.rolling(_ATR_PERIOD).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    for w in [7, _RSI_PERIOD]:
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

    for lag in [1, 2, 3, 5]:
        df[f"ret_{lag}"] = np.log(c / c.shift(lag))

    df["vol_norm"] = vol / (vol.rolling(20).mean() + 1e-9)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    return df


# ──────────────────────────────────────────────────────────────
# Паттерны Метки
# ──────────────────────────────────────────────────────────────

def build_metka_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"]
    o   = df["open"]
    h   = df["high"]
    lo  = df["low"]
    atr = df["atr_20"]
    rsi = df[f"rsi_{_RSI_PERIOD}"]
    ef  = df[f"ema_{_EMA_FAST}"]
    es  = df[f"ema_{_EMA_SLOW}"]

    rng      = h - lo
    body     = c - o
    abs_body = body.abs()
    upper_w  = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_w  = pd.concat([o, c], axis=1).min(axis=1) - lo
    mid      = (h + lo) / 2.0

    # ── Method 2: Pin Bar
    big_bar = (rng > atr * _SPIKE_ATR) & (rng > 0)
    df["pin_bar_buy"]  = (big_bar & (lower_w > rng * 0.5) & (c > mid)).astype(float)
    df["pin_bar_sell"] = (big_bar & (upper_w > rng * 0.5) & (c < mid)).astype(float)

    # ── Method 3: Momentum Reversal
    prior_body = body.shift(1)
    prior_mom  = prior_body.rolling(_REV_LB).sum()
    bear_count = (prior_body < 0).rolling(_REV_LB).sum()
    bull_count = (prior_body > 0).rolling(_REV_LB).sum()
    avg_body   = prior_body.abs().rolling(_REV_LB).mean()

    strong_buy_bar  = (abs_body > atr * 0.3) & (abs_body > avg_body) & \
                      (rng > 0) & (abs_body > rng * 0.4) & (body > 0)
    strong_sell_bar = (abs_body > atr * 0.3) & (abs_body > avg_body) & \
                      (rng > 0) & (abs_body > rng * 0.4) & (body < 0)

    df["momentum_rev_buy"]  = (
        (bear_count >= 3) & (prior_mom < -atr * 0.7) & strong_buy_bar
    ).astype(float)
    df["momentum_rev_sell"] = (
        (bull_count >= 3) & (prior_mom >  atr * 0.7) & strong_sell_bar
    ).astype(float)

    # ── Method 4: Squeeze Breakout
    avg_range_6 = (h - lo).shift(1).rolling(_SQZ_LB).mean()
    squeezed    = avg_range_6 < atr * 0.5
    big_body    = (rng > atr) & (abs_body > rng * 0.5)
    trend_up    = ef > es
    trend_dn    = ef < es

    df["squeezed"]            = squeezed.astype(float)
    df["squeeze_break_buy"]   = (squeezed & big_body & (body > 0) & trend_up).astype(float)
    df["squeeze_break_sell"]  = (squeezed & big_body & (body < 0) & trend_dn).astype(float)

    # ── Method 5: RSI Divergence (упрощённая)
    # Бычья: цена у нового минимума, RSI НЕ у минимума (выше на 3+)
    min_lo_30 = lo.rolling(_DIV_LB).min().shift(_DIV_SW)
    min_rsi_30 = rsi.rolling(_DIV_LB).min().shift(_DIV_SW)
    price_near_low  = lo <= min_lo_30 + atr * 0.5
    rsi_not_at_low  = rsi > min_rsi_30 + 3.0

    max_hi_30 = h.rolling(_DIV_LB).max().shift(_DIV_SW)
    max_rsi_30 = rsi.rolling(_DIV_LB).max().shift(_DIV_SW)
    price_near_high  = h >= max_hi_30 - atr * 0.5
    rsi_not_at_high  = rsi < max_rsi_30 - 3.0

    df["rsi_div_buy"]  = (price_near_low  & rsi_not_at_low  & (body > 0)).astype(float)
    df["rsi_div_sell"] = (price_near_high & rsi_not_at_high & (body < 0)).astype(float)

    # ── Swing Filter (по умолчанию включён)
    lowest_low_sw   = lo.rolling(_SWING_BARS).min().shift(1)
    highest_high_sw = h.rolling(_SWING_BARS).max().shift(1)
    swing_ok_buy    = lo <= lowest_low_sw  + atr * _SWING_ATR
    swing_ok_sell   = h  >= highest_high_sw - atr * _SWING_ATR

    df["swing_near_low"]  = swing_ok_buy.astype(float)
    df["swing_near_high"] = swing_ok_sell.astype(float)

    # ── Итоговый сигнал (любой паттерн + swing-фильтр)
    any_raw_buy  = (df["pin_bar_buy"].astype(bool)  |
                    df["momentum_rev_buy"].astype(bool)  |
                    df["squeeze_break_buy"].astype(bool)  |
                    df["rsi_div_buy"].astype(bool))
    any_raw_sell = (df["pin_bar_sell"].astype(bool) |
                    df["momentum_rev_sell"].astype(bool) |
                    df["squeeze_break_sell"].astype(bool) |
                    df["rsi_div_sell"].astype(bool))

    df["metka_buy"]  = (any_raw_buy  & swing_ok_buy).astype(float)
    df["metka_sell"] = (any_raw_sell & swing_ok_sell).astype(float)

    # ── Количество одновременно сработавших паттернов (сила сигнала)
    df["metka_buy_strength"]  = (
        df["pin_bar_buy"] + df["momentum_rev_buy"] +
        df["squeeze_break_buy"] + df["rsi_div_buy"]
    )
    df["metka_sell_strength"] = (
        df["pin_bar_sell"] + df["momentum_rev_sell"] +
        df["squeeze_break_sell"] + df["rsi_div_sell"]
    )

    # ── Сильный сигнал (squeeze или большой бар — как в Metka)
    df["is_strong"] = (squeezed | (rng > atr * 1.8)).astype(float)

    # ── EMA тренд
    df["trend_up"] = trend_up.astype(float)
    df["trend_dn"] = trend_dn.astype(float)
    df["ema_dist"] = (ef - es) / (atr + 1e-9)   # нормированное расстояние между EMA

    return df


# ──────────────────────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = build_base_features(df)
    df = build_metka_features(df)
    return df
