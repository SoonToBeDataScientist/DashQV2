from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

SNAPSHOT_COLS = ["pc_vol", "pc_oi", "atm_iv", "iv_skew", "short_ratio",
                 "short_pct_float", "btc_dom"]


def _yf_close(tick, start) -> pd.Series:
    try:
        import yfinance as yf
        df = yf.download(tick, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"] if "Close" in df else df["close"]
        s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception:
        return pd.Series(dtype=float)


# ---- options-market history (backtestable immediately: ^VIX ^VXV ^VVIX ^SKEW) ----
def options_market_history(start) -> pd.DataFrame:
    df = pd.DataFrame({k: _yf_close(t, start) for k, t in
                       {"vix": "^VIX", "vxv": "^VXV", "vvix": "^VVIX", "skew": "^SKEW"}.items()}).ffill()
    if df.empty or df["vix"].dropna().empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    vxv = df["vxv"].replace(0, np.nan).fillna(df["vix"])
    out["vix_ts"] = df["vix"] / vxv - 1                    # term structure: >0 = backwardation
    out["vix_ts_chg"] = out["vix_ts"].diff(5)
    out["skew_lvl"] = df["skew"] / 100 - 1.2               # SKEW ~120 baseline -> ~0
    out["skew_chg"] = df["skew"].diff(5) / 100
    out["vvix_lvl"] = df["vvix"] / 100
    return out.ffill()


# ---- on-chain history (keyless, backtestable) ----
def fear_greed_history() -> pd.Series:
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 0}, timeout=15)
        s = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s").normalize():
                       float(d["value"]) for d in r.json().get("data", [])})
        return s.sort_index() / 50 - 1                     # 0..100 -> -1..+1
    except Exception:
        return pd.Series(dtype=float)


def _bc_chart(name, timespan="3y") -> pd.Series:
    try:
        r = requests.get(f"https://api.blockchain.info/charts/{name}",
                         params={"timespan": timespan, "format": "json", "sampled": "true"},
                         timeout=20)
        return pd.Series({pd.to_datetime(v["x"], unit="s").normalize(): float(v["y"])
                          for v in r.json().get("values", [])}).sort_index()
    except Exception:
        return pd.Series(dtype=float)


def onchain_history() -> pd.DataFrame:
    out = pd.DataFrame()
    fng = fear_greed_history()
    if not fng.empty:
        out["fng"], out["fng_chg5"] = fng, fng.diff(5)
    hr = _bc_chart("hash-rate")
    if not hr.empty:
        out["hashr_chg21"] = hr.pct_change(21)
    tx = _bc_chart("n-transactions")
    if not tx.empty:
        out["ntx_chg21"] = tx.pct_change(21)
    return out.ffill() if not out.empty else out


# ---- point-in-time snapshots (accumulated daily by the scheduler/pipeline) ----
def symbol_snapshot(sym: str, asset_class: str) -> dict:
    row: dict = {"symbol": sym}
    try:
        import yfinance as yf
        t = yf.Ticker(sym.replace("/", "-") if asset_class == "crypto" else sym)
        if asset_class == "stock":
            exps = getattr(t, "options", ()) or ()
            if exps:
                ch = t.option_chain(exps[0])
                cv, pv = ch.calls["volume"].sum(), ch.puts["volume"].sum()
                coi, poi = ch.calls["openInterest"].sum(), ch.puts["openInterest"].sum()
                row["pc_vol"] = float(pv / cv) if cv else np.nan
                row["pc_oi"] = float(poi / coi) if coi else np.nan
                hist = t.history(period="5d")
                spot = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
                if spot == spot:
                    atm = pd.concat([ch.calls, ch.puts])
                    atm = atm.iloc[(atm["strike"] - spot).abs().argsort()[:4]]
                    row["atm_iv"] = float(atm["impliedVolatility"].mean())
                    op = ch.puts[(ch.puts["strike"] < spot * 0.97) & (ch.puts["strike"] > spot * 0.85)]
                    oc = ch.calls[(ch.calls["strike"] > spot * 1.03) & (ch.calls["strike"] < spot * 1.15)]
                    if not op.empty and not oc.empty:
                        row["iv_skew"] = float(op["impliedVolatility"].mean()
                                               - oc["impliedVolatility"].mean())
            info = t.info or {}
            if info.get("shortRatio") is not None:
                row["short_ratio"] = float(info["shortRatio"])
            if info.get("shortPercentOfFloat") is not None:
                row["short_pct_float"] = float(info["shortPercentOfFloat"])
    except Exception:
        pass
    return row


def btc_dominance() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        return float(r.json()["data"]["market_cap_percentage"]["btc"]) / 50 - 1
    except Exception:
        return np.nan


def load_snapshots(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "symbol"] + SNAPSHOT_COLS)


def collect_snapshots(symbols, asset_map, path: str) -> pd.DataFrame:
    """Append today's per-symbol snapshots; builds genuine forward history over time."""
    rows = []
    bd = btc_dominance()
    for sym in symbols:
        r = symbol_snapshot(sym, asset_map.get(sym, "stock"))
        r["symbol"], r["btc_dom"] = sym, bd
        rows.append(r)
    new = pd.DataFrame(rows)
    new.insert(0, "date", pd.Timestamp.utcnow().normalize())
    all_ = pd.concat([load_snapshots(path), new], ignore_index=True)
    all_["date"] = pd.to_datetime(all_["date"]).dt.normalize()
    all_ = all_.drop_duplicates(subset=["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    all_.to_csv(path, index=False)
    return all_


def get_exo(start, snapshot_path: str) -> dict:
    return {"options_market": options_market_history(start),
            "onchain": onchain_history(),
            "snapshots": load_snapshots(snapshot_path)}
