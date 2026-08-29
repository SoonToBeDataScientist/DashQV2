from __future__ import annotations

import json

import pandas as pd


def signal_events(journal_df) -> pd.DataFrame:
    """Parse journal into (ts, symbol, signal, price?) rows."""
    rows = []
    for _, r in journal_df.iterrows():
        if r["kind"] not in ("signals", "realtime"):
            continue
        try:
            payload = json.loads(r["payload"])
        except Exception:
            continue
        ts = pd.Timestamp(r["ts"]).tz_localize(None)
        if r["kind"] == "signals":                       # {sym: value}
            for sym, v in payload.items():
                rows.append({"ts": ts, "symbol": sym, "signal": float(v)})
        else:                                            # {sym: {signal, price}}
            for sym, d in payload.items():
                if isinstance(d, dict) and "signal" in d:
                    rows.append({"ts": ts, "symbol": sym, "signal": float(d["signal"]),
                                 "price": d.get("price")})
    return pd.DataFrame(rows)


def report(events, closes, horizon=5, rolling=250) -> dict:
    """Live forward test: do recorded signals predict subsequent returns?"""
    if events is None or events.empty:
        return {}
    ev = events.dropna(subset=["signal"]).copy()
    ev["date"] = ev["ts"].dt.normalize()
    ev = ev.groupby(["date", "symbol"], as_index=False)["signal"].mean()
    recs, closes = [], closes.sort_index()
    for r in ev.itertuples():
        if r.symbol not in closes:
            continue
        px = closes[r.symbol].dropna()
        past, fut = px[px.index <= r.date], px[px.index > r.date]
        if past.empty or len(fut) < horizon:
            continue
        recs.append({"date": r.date, "symbol": r.symbol, "signal": r.signal,
                     "fwd_ret": float(fut.iloc[horizon - 1] / past.iloc[-1] - 1)})
    df = pd.DataFrame(recs)
    if len(df) < 10:
        return {"n": len(df), "detail": df}
    df = df.sort_values("date")
    df["bucket"] = pd.cut(df["signal"], [-1.01, -0.33, 0.33, 1.01],
                          labels=["short", "flat", "long"])
    d = df.copy()
    d["sr"], d["rr"] = d["signal"].rank(), d["fwd_ret"].rank()
    d["ic_roll"] = d["sr"].rolling(rolling, min_periods=30).corr(d["rr"])
    return {
        "n": len(df),
        "ic": float(df["signal"].corr(df["fwd_ret"], method="spearman")),
        "by_bucket": df.groupby("bucket", observed=True)["fwd_ret"].agg(["mean", "count"]),
        "rolling": d[["date", "ic_roll"]].dropna(),
        "edge": (df["signal"] * df["fwd_ret"]).groupby(df["date"]).mean(),
        "detail": df,
    }

