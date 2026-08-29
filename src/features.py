from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "trend": ["sma_ratio", "ema_ratio", "adx_14"],
    "momentum": ["ret_1", "ret_5", "ret_21", "mom_10", "rsi_14", "macd_hist", "stoch_k"],
    "volatility": ["vol_20", "atr_pct", "bb_z"],
    "volume": ["vol_ratio", "obv_slope"],
    "options": ["vix_ts", "vix_ts_chg", "skew_lvl", "skew_chg", "vvix_lvl",
                "pc_vol", "pc_oi", "atm_iv", "iv_skew"],
    "short": ["short_ratio", "short_pct_float"],
    "onchain": ["fng", "fng_chg5", "hashr_chg21", "ntx_chg21", "btc_dom"],
}
MACRO_FEATURES = ["vix_lvl", "vix_chg5", "vix_z", "y10_chg5", "yc_spread", "spy_ret5",
                  "spy_ret21", "oil_ret5", "gold_ret5", "dxy_chg5", "unrate", "cons_sent"]
SENT_FEATURES = ["sent", "sent_chg3", "sent_z", "news_cnt"]
HORIZONS = (3, 5, 10)


def feature_columns(genome) -> list:
    cols = []
    for g in genome.feature_groups:
        cols += FEATURE_GROUPS.get(g, [])
    if genome.use_macro:
        cols += MACRO_FEATURES
    if genome.use_sentiment:
        cols += SENT_FEATURES
    return cols


# ---------------------------------------------------------------- indicators
def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _macd_hist(c, fast=12, slow=26, sig=9):
    m = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    return (m - m.ewm(span=sig, adjust=False).mean()) / c


def _stoch(h, l, c, n=14):
    ll, hh = l.rolling(n).min(), h.rolling(n).max()
    return (c - ll) / (hh - ll).replace(0, np.nan)


def _true_range(h, l, c):
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def _adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    atr = _true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = c.pct_change()
    f["ret_5"] = c.pct_change(5)
    f["ret_21"] = c.pct_change(21)
    f["mom_10"] = c / c.shift(10) - 1
    f["rsi_14"] = _rsi(c) / 100 - 0.5
    f["macd_hist"] = _macd_hist(c)
    f["stoch_k"] = _stoch(h, l, c) - 0.5
    f["adx_14"] = _adx(h, l, c) / 100
    f["sma_ratio"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    f["ema_ratio"] = c.ewm(span=12, adjust=False).mean() / c.ewm(span=48, adjust=False).mean() - 1
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["atr_pct"] = _true_range(h, l, c).rolling(14).mean() / c
    f["bb_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    v20 = v.rolling(20).mean()
    f["vol_ratio"] = v / v20 - 1
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = obv.diff(5) / (v20 * 5 + 1)
    return f


# ---------------------------------------------------------------- assembly
def _reindex_ffill(df: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    """Align external daily series to idx, forward-filling across missing dates
    (works even when idx has a single row, e.g. the live bar)."""
    if df is None or df.empty:
        return df
    return df.reindex(df.index.union(idx)).ffill().reindex(idx)


def _join_extras(f, sym, macro, sent, genome, exo=None):
    if genome.use_macro and macro is not None and not macro.empty:
        f = f.join(_reindex_ffill(macro, f.index))
    if genome.use_sentiment and sent is not None and not sent.empty:
        s = pd.DataFrame(index=f.index)
        cs, cn = f"{sym}__sent", f"{sym}__news"
        bse = sent[cs].reindex(f.index).fillna(0.0) if cs in sent else pd.Series(0.0, index=f.index)
        s["sent"] = bse
        s["sent_chg3"] = bse - bse.shift(3)
        s["sent_z"] = (bse - bse.rolling(20, 5).mean()) / bse.rolling(20, 5).std()
        s["news_cnt"] = sent[cn].reindex(f.index).fillna(0.0) if cn in sent else 0.0
        f = f.join(s)
    if exo:
        for key in ("options_market", "onchain"):
            df = exo.get(key)
            if df is not None and not df.empty:
                f = f.join(_reindex_ffill(df, f.index))
        snap = exo.get("snapshots")
        if snap is not None and not snap.empty:
            s = snap[snap["symbol"] == sym].drop(columns=["symbol"], errors="ignore")
            if not s.empty:
                s = s.set_index(pd.to_datetime(s["date"])).drop(columns=["date"], errors="ignore")
                s = s[~s.index.duplicated(keep="last")].sort_index()
                f = f.join(_reindex_ffill(s, f.index))
    return f


def build_panel(bars: dict, macro, sent, genome, exo=None) -> pd.DataFrame:
    """Stacked panel: one row per (date, symbol) with features + forward-return targets."""
    frames = []
    for sym, df in bars.items():
        f = technical_features(df)
        f = _join_extras(f, sym, macro, sent, genome, exo)
        f["close"] = df["close"]
        for h in HORIZONS:
            f[f"target_{h}"] = df["close"].shift(-h) / df["close"] - 1
        f["symbol"] = sym
        f["date"] = f.index
        frames.append(f.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def build_live_row(sym, daily, intraday, macro, sent, genome, exo=None) -> pd.DataFrame:
    """Append a synthetic 'today' bar from intraday data, then compute features."""
    df = daily.copy()
    if intraday is not None and not intraday.empty:
        d = intraday.index[-1].normalize()
        today = intraday[intraday.index.normalize() == d]
        if not today.empty:
            df = df[df.index < d]
            df.loc[d] = {"open": today["open"].iloc[0], "high": today["high"].max(),
                         "low": today["low"].min(), "close": today["close"].iloc[-1],
                         "volume": today["volume"].sum()}
    f = technical_features(df)
    f = _join_extras(f, sym, macro, sent, genome, exo)
    return f.iloc[[-1]]
