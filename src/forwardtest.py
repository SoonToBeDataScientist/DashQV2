from __future__ import annotations

import json

import numpy as np
import pandas as pd


def signal_events(journal_df, kinds=("signals", "realtime")) -> pd.DataFrame:
    """Parse journal into (ts, symbol, signal, price?) rows."""
    rows = []
    for _, r in journal_df.iterrows():
        if r["kind"] not in kinds:
            continue
        try:
            payload = json.loads(r["payload"])
        except Exception:
            continue
        ts = pd.Timestamp(r["ts"]).tz_localize(None)
        if r["kind"] == "signals":                       # {sym: value}
            for sym, v in payload.items():
                if v is None:                             # e.g. weekend: equity has no fresh bar
                    continue
                try:
                    rows.append({"ts": ts, "symbol": sym, "signal": float(v)})
                except (TypeError, ValueError):
                    continue
        else:                                            # {sym: {signal, price}}
            for sym, d in payload.items():
                if isinstance(d, dict) and d.get("signal") is not None:
                    try:
                        rows.append({"ts": ts, "symbol": sym, "signal": float(d["signal"]),
                                     "price": d.get("price")})
                    except (TypeError, ValueError):
                        continue
    return pd.DataFrame(rows)


def report(events, closes, horizon=5, rolling=250, session_cfg=None) -> dict:
    """Live forward test: do recorded signals predict subsequent returns?
    session_cfg: universe config whose "session" decides which trading day a run belongs to —
    a run logged at 00:40 UTC is still the previous US session, so its base price must be that
    session's close, not the next one's. None = plain UTC calendar date."""
    if events is None or events.empty:
        return {}
    ev = events.dropna(subset=["signal"]).copy()
    if session_cfg is None:
        ev["date"] = ev["ts"].dt.normalize()
    else:
        from .session import session_dates
        ev["date"] = session_dates(session_cfg, ev["ts"]).values
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


# ---------------------------------------------------------------- forward book (sized portfolio)
def book_events(journal_df, session_cfg=None) -> pd.DataFrame:
    """The weights the pipeline decided, as a (session date x symbol) frame from the journal's
    "weights" events. NaN = that symbol wasn't tradable that run, so its position was left as it
    was. If a session was run more than once (a manual run plus the scheduled one) the LAST run
    wins. session_cfg works as in report(): a 00:40 UTC run belongs to the previous US session."""
    recs = []
    for _, r in journal_df.iterrows():
        if r["kind"] != "weights":
            continue
        try:
            ws = (json.loads(r["payload"]).get("weights") or {})
        except Exception:
            continue
        row = {}
        for sym, v in ws.items():
            try:
                row[sym] = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                row[sym] = float("nan")
        if row:
            recs.append((pd.Timestamp(r["ts"]).tz_localize(None), row))
    if not recs:
        return pd.DataFrame()
    recs.sort(key=lambda t: t[0])
    ts = pd.Series([t for t, _ in recs])
    if session_cfg is None:
        dates = ts.dt.normalize()
    else:
        from .session import session_dates
        dates = session_dates(session_cfg, ts)
    df = pd.DataFrame([row for _, row in recs])
    df.index = pd.DatetimeIndex(dates.values)
    return df[~df.index.duplicated(keep="last")].sort_index()


def book_report(book, closes, opens=None, exec_open=(), fee_bps=2.0, slippage_bps=5.0) -> dict:
    """Live forward test of the SIZED portfolio: what the journaled weights earned on the prices
    that followed, scored by backtest.evaluate_weights — the same arithmetic as the backtest
    (next-open fills for `exec_open` stocks, the same costs). The newest decision has no realized
    return yet, so it counts from the next session."""
    if book is None or book.empty:
        return {"n": 0}
    n = len(book)
    if n < 3:
        return {"n": n}
    from .backtest import evaluate_weights
    px = closes.sort_index().ffill()
    px = px[px.index >= book.index.min()]
    if len(px) < 3:
        return {"n": n}
    res = evaluate_weights(px, book, fee_bps, slippage_bps, opens=opens, exec_open=exec_open)
    bench = (1 + px.pct_change().mean(axis=1).fillna(0)).cumprod()
    return {"n": n, "equity": res.equity, "drawdown": res.drawdown,
            "gross": res.weights.abs().sum(axis=1), "metrics": res.metrics, "bench": bench}


# ---------------------------------------------------------------- sentiment shadow (live A/B)
def shadow_events(journal_df, base=None) -> pd.DataFrame:
    """(ts, symbol, signal, uses_sentiment) rows from the journal's "shadow" events: what the
    shadow twin (the champion with news sentiment flipped, never traded) said each run. `base`
    keeps only runs where the sitting champion had that name, so a promotion starts a fresh test."""
    rows = []
    for _, r in journal_df.iterrows():
        if r["kind"] != "shadow":
            continue
        try:
            p = json.loads(r["payload"])
        except Exception:
            continue
        if base is not None and p.get("base") != base:
            continue
        ts = pd.Timestamp(r["ts"]).tz_localize(None)
        for sym, v in (p.get("signals") or {}).items():
            try:
                if v is not None:
                    rows.append({"ts": ts, "symbol": sym, "signal": float(v),
                                 "uses_sentiment": bool(p.get("uses_sentiment"))})
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(rows)


def shadow_report(journal_df, closes, horizon=5, session_cfg=None, base=None,
                  min_obs=60, min_t=2.0) -> dict:
    """Live, out-of-sample A/B of news sentiment. Every pipeline run journals the strategy's real
    signals AND the shadow twin's (same strategy, sentiment flipped, not traded). Here both are
    scored against the same realized forward returns, session by session. edge = (shadow - real
    signal) x forward return, averaged over symbols per session, then over blocks of `horizon`
    sessions and t-tested (forward returns overlap, so single sessions aren't independent).
    `better` = at least `min_obs` matured sessions, t >= min_t, and a higher rank IC — meaning the
    SHADOW is the better variant (check `shadow_uses_sentiment` to know which way that points)."""
    sh = shadow_events(journal_df, base)
    if sh.empty:
        return {"n": 0}
    a = report(signal_events(journal_df, ("signals",)), closes, horizon, session_cfg=session_cfg).get("detail")
    b = report(sh, closes, horizon, session_cfg=session_cfg).get("detail")
    if a is None or b is None or a.empty or b.empty:
        return {"n": 0}
    m = a.merge(b, on=["date", "symbol"], suffixes=("_main", "_shadow"))
    if m.empty:
        return {"n": 0}
    out = {"n": int(m["date"].nunique()), "shadow_uses_sentiment": bool(sh["uses_sentiment"].iloc[-1]),
           "ic_main": float(m["signal_main"].corr(m["fwd_ret_main"], method="spearman")),
           "ic_shadow": float(m["signal_shadow"].corr(m["fwd_ret_main"], method="spearman"))}
    d = ((m["signal_shadow"] - m["signal_main"]) * m["fwd_ret_main"]).groupby(m["date"]).mean().sort_index()
    B = max(5, int(horizon))
    blocks = d.groupby(np.arange(len(d)) // B).mean()
    if len(d) % B < B // 2 and len(blocks) > 1:
        blocks = blocks.iloc[:-1]                        # drop a stub final block
    t = (float(blocks.mean() / (blocks.std(ddof=1) / np.sqrt(len(blocks)) + 1e-12))
         if len(blocks) >= 6 else float("nan"))
    out.update({"edge_t": t, "blocks": int(len(blocks)), "min_obs": int(min_obs), "min_t": float(min_t),
                "better": bool(out["n"] >= min_obs and t == t and t >= min_t
                               and out["ic_shadow"] > out["ic_main"])})
    return out
