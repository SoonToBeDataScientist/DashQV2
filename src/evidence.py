"""Is there a statistically real edge? — tests that can say "no".

Everything here is pure (numpy/pandas, no network) so the same code judges the live journal in
the daily pipeline, the walk-forward backtest in the app, and the synthetic worlds in tests/.

Why not just look at the IC or the Sharpe?
  * Forward returns overlap (a 3-day return is reported every day), so a few weeks of signals are
    a handful of independent observations, not hundreds. A plain t-test or a pooled correlation
    over-states them. Every test below uses overlap-robust errors (Newey-West), a circular block
    bootstrap, or a permutation that keeps the autocorrelation intact.
  * A long-only book earns money in a rising market for free. "The traded book made money" is
    not an edge; "it made more than the same exposure with random timing" is. The permutation
    test shifts the whole signal history against the returns (keeping the exposure profile and
    the cross-asset structure) and asks how often that does as well. The excess/alpha tests
    compare with simply holding.
  * The lab keeps the best of many genomes on one window, so its winner's Sharpe is inflated by
    selection. deflated_sharpe() discounts it for the number of genomes tried.
  * A p-value says nothing about how long you'd need to wait. days_to_significance() and
    min_detectable() say how much data it takes to see an effect of a given size, so "no
    evidence yet" can be told apart from "no edge".
"""
from __future__ import annotations

import math
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER = 0.5772156649015329

LEVELS = {
    "too_early": "Too early to say",
    "none": "No evidence of an edge",
    "hint": "A hint, not distinguishable from luck",
    "evidence": "Evidence of an edge",
    "strong": "Strong evidence of an edge",
}


# ------------------------------------------------------------------ small tools
def periods_per_year(index) -> int:
    """365 if the series has weekend rows (crypto in the book), else 252."""
    idx = pd.DatetimeIndex(index)
    return 365 if len(idx) and bool((idx.dayofweek >= 5).any()) else 252


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def nw_mean_test(x, lags: int | None = None) -> dict:
    """Mean of x with a Newey-West (Bartlett) standard error. `p_pos` is one-sided: the chance of
    a mean this large if the true mean were <= 0. `n_eff` is how many independent observations
    the series is worth once its autocorrelation is accounted for."""
    x = _finite(x)
    n = len(x)
    if n < 5:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "p_pos": float("nan"), "n_eff": float(n)}
    m = float(x.mean())
    e = x - m
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    lags = max(0, min(int(lags), n - 2))
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    s = max(s, 1e-18)
    se = math.sqrt(s / n)
    t = m / se
    return {"n": n, "mean": m, "se": se, "t": t, "p_pos": 1.0 - _N.cdf(t),
            "n_eff": float(min(n, max(1.0, n * g0 / s)))}


def boot_mean_ci(x, block: int = 5, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Circular block bootstrap CI for the mean (blocks keep the autocorrelation)."""
    x = _finite(x)
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    block = max(1, min(int(block), n))
    nb = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ------------------------------------------------------------------ signals -> (date, symbol) table
def signal_detail(signals: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """(date, symbol, signal, fwd_ret) from a signal matrix and daily closes. fwd_ret is the
    return from the last close on/before the signal date to the close `horizon` sessions of that
    symbol later — the same definition forwardtest.report uses for the live journal, so live and
    walk-forward evidence are directly comparable. Dates without enough future are dropped."""
    recs = []
    closes = closes.sort_index()
    for sym in signals.columns:
        if sym not in closes:
            continue
        px = closes[sym].dropna()
        if px.empty:
            continue
        s = signals[sym].dropna()
        if s.empty:
            continue
        pos = px.index.searchsorted(s.index, side="right")          # entries on/before each date
        base_i, fut_i = pos - 1, pos + horizon - 1
        ok = (base_i >= 0) & (fut_i < len(px))
        if not ok.any():
            continue
        v = px.to_numpy()
        fwd = v[fut_i[ok]] / v[base_i[ok]] - 1.0
        recs.append(pd.DataFrame({"date": s.index[ok], "symbol": sym,
                                  "signal": s.to_numpy()[ok], "fwd_ret": fwd}))
    if not recs:
        return pd.DataFrame(columns=["date", "symbol", "signal", "fwd_ret"])
    return pd.concat(recs, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _wide(detail: pd.DataFrame, value: str) -> pd.DataFrame:
    return detail.pivot_table(index="date", columns="symbol", values=value, aggfunc="mean").sort_index()


# ------------------------------------------------------------------ the statistics
def cross_section_ic(detail: pd.DataFrame, min_assets: int = 4) -> pd.Series:
    """Per-date Spearman IC across symbols: did the model rank the assets right on that date?"""
    out = {}
    for d, g in detail.dropna(subset=["signal", "fwd_ret"]).groupby("date"):
        if len(g) >= min_assets and g["signal"].nunique() > 1 and g["fwd_ret"].nunique() > 1:
            out[d] = float(g["signal"].corr(g["fwd_ret"], method="spearman"))
    return pd.Series(out, dtype=float).sort_index()


def _rank_z(M: pd.DataFrame) -> np.ndarray:
    """Per-column rank, standardised — a robust stand-in for the raw value in a permutation."""
    R = M.rank(axis=0)
    Z = (R - R.mean()) / R.std(ddof=0).replace(0, np.nan)
    return Z.to_numpy(dtype=float)


def _asset_mean(P: np.ndarray) -> float:
    """Mean over assets of each asset's mean product. Equal weight per asset, so a shift that moves
    a stock's weekend gaps onto weekdays changes how many cells an asset has, not how much it counts
    (crypto and stocks have very different vol and drift)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols = np.nanmean(P, axis=0)
        return float(np.nanmean(cols)) if np.isfinite(cols).any() else float("nan")


def _perm_p(A: np.ndarray, B: np.ndarray, min_shift: int, max_shifts: int, seed: int):
    """One-sided permutation p for stat = mean over assets of mean(A * B), with A (signals)
    circularly shifted in time against B (returns). A joint shift of every asset keeps each
    series' autocorrelation, the exposure profile and the cross-asset structure — only the
    timing relation is broken. Shifts closer than `min_shift` rows to 0 (or to a full turn) are
    excluded because they still overlap the real alignment. Returns (observed, p, n_shifts,
    null_values)."""
    T = A.shape[0]
    obs = _asset_mean(A * B)
    ks = np.arange(min_shift, T - min_shift + 1)
    if len(ks) == 0 or not np.isfinite(obs):
        return obs, float("nan"), 0, np.array([])
    if len(ks) > max_shifts:
        ks = np.random.default_rng(seed).choice(ks, size=max_shifts, replace=False)
    null = np.array([_asset_mean(np.roll(A, int(k), axis=0) * B) for k in ks])
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return obs, float("nan"), 0, null
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return obs, float(p), int(len(null)), null


def edge_report(*args, **kwargs) -> dict:
    """See _edge_report. Wrapped so degenerate inputs (a constant signal, an empty asset) give a
    verdict instead of numpy warnings in the pipeline log."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _edge_report(*args, **kwargs)


def _edge_report(detail: pd.DataFrame, horizon: int = 5, long_only: bool = True,
                 n_boot: int = 2000, max_shifts: int = 2000, seed: int = 0,
                 min_eff: float = 30.0) -> dict:
    """Does the model's timing/selection earn more than chance? `detail` is (date, symbol,
    signal, fwd_ret). Headline test = the TRADED book's edge (signal x forward return, long-only
    if the account is) against the same signals shifted in time. Diagnostics: per-date
    cross-sectional IC (which asset), time-series rank IC (when), and the old pooled IC for
    comparison with what the dashboard showed before."""
    d = detail.dropna(subset=["signal", "fwd_ret"]) if detail is not None else pd.DataFrame()
    n_dates = int(d["date"].nunique()) if len(d) else 0
    base = {"n_dates": n_dates, "n_obs": int(len(d)), "horizon": int(horizon), "long_only": bool(long_only)}
    if n_dates < 10:
        return {**base, "level": "too_early", "label": LEVELS["too_early"], "n_eff": float(n_dates)}
    S = _wide(d, "signal")
    R = _wide(d, "fwd_ret").reindex(index=S.index, columns=S.columns)
    S_eff = S.clip(lower=0.0) if long_only else S
    block = max(int(horizon), 3)

    edge_t = (S_eff * R).mean(axis=1, skipna=True).dropna()            # per-date traded-book edge
    nw = nw_mean_test(edge_t.to_numpy(), lags=max(int(horizon) - 1, 0))
    lo, hi = boot_mean_ci(edge_t.to_numpy(), block, n_boot, seed)
    obs, p_perm, n_shifts, null = _perm_p(S_eff.to_numpy(float), R.to_numpy(float),
                                          min_shift=2 * int(horizon), max_shifts=max_shifts, seed=seed)
    # drift-free reference: what the same exposure earns with random timing
    null_mean = float(null.mean()) if len(null) else float("nan")
    edge = {"mean": nw["mean"], "t": nw["t"], "p_nw": nw["p_pos"], "ci": (lo, hi),
            "p_perm": p_perm, "n_shifts": n_shifts, "random_timing_mean": null_mean,
            "above_random_timing": (obs - null_mean) if np.isfinite(null_mean) else float("nan")}

    cs = cross_section_ic(d)
    cs_nw = nw_mean_test(cs.to_numpy(), lags=max(int(horizon) - 1, 0))
    cs_lo, cs_hi = boot_mean_ci(cs.to_numpy(), block, n_boot, seed + 1)
    ic_cs = {"mean": cs_nw["mean"], "t": cs_nw["t"], "p_nw": cs_nw["p_pos"], "ci": (cs_lo, cs_hi),
             "n_dates": int(len(cs)), "sd": float(cs.std(ddof=1)) if len(cs) > 2 else float("nan"),
             "n_eff": cs_nw["n_eff"]}

    ZS, ZR = _rank_z(S_eff if long_only else S), _rank_z(R)
    ts_obs, ts_p, _, _ = _perm_p(ZS, ZR, 2 * int(horizon), max_shifts, seed + 2)
    ic_ts = {"mean": ts_obs, "p_perm": ts_p}

    pooled = float(d["signal"].corr(d["fwd_ret"], method="spearman"))
    n_eff = float(nw["n_eff"])
    level = "too_early"
    if n_eff >= min_eff:
        if p_perm == p_perm and p_perm < 0.01 and lo > 0:
            level = "strong"
        elif p_perm == p_perm and p_perm < 0.05 and lo > 0:
            level = "evidence"
        elif p_perm == p_perm and p_perm < 0.20:
            level = "hint"
        else:
            level = "none"
    return {**base, "level": level, "label": LEVELS[level], "n_eff": n_eff, "min_eff": float(min_eff),
            "edge": edge, "ic_cs": ic_cs, "ic_ts": ic_ts, "ic_pooled": pooled,
            "need": days_to_significance(nw["t"], n_eff, base["n_dates"]),
            "detectable": min_detectable(float(edge_t.std(ddof=1)), n_eff)}


# ------------------------------------------------------------------ vs holding
def excess_report(strat_ret: pd.Series, hold_ret: pd.Series, block: int = 21,
                  n_boot: int = 2000, seed: int = 0, min_eff: float = 30.0) -> dict:
    """Did the strategy beat holding, beyond what its market exposure explains? Both inputs are
    daily returns of books sized the same way. Reports the excess return (strategy - hold), the
    CAPM-style alpha against hold, and a paired block-bootstrap of the Sharpe difference."""
    df = pd.concat({"s": strat_ret, "h": hold_ret}, axis=1).dropna()
    n = len(df)
    base = {"n": n}
    if n < 20:
        return {**base, "level": "too_early", "label": LEVELS["too_early"]}
    ann = periods_per_year(df.index)
    s, h = df["s"].to_numpy(), df["h"].to_numpy()
    d = s - h
    nw = nw_mean_test(d)
    lo, hi = boot_mean_ci(d, block, n_boot, seed)
    var_h = float(np.var(h))
    beta = float(np.cov(s, h, ddof=0)[0, 1] / var_h) if var_h > 0 else float("nan")
    resid = s - beta * h if beta == beta else s
    a = nw_mean_test(resid)
    a_lo, a_hi = boot_mean_ci(resid, block, n_boot, seed + 1)

    def sharpe(x):
        sd = x.std(axis=-1, ddof=1)
        return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1.0), 0.0) * math.sqrt(ann)

    blk = max(1, min(int(block), n))
    nb = int(math.ceil(n / blk))
    rng = np.random.default_rng(seed + 2)
    idx = ((rng.integers(0, n, size=(n_boot, nb))[:, :, None] + np.arange(blk)[None, None, :]) % n
           ).reshape(n_boot, -1)[:, :n]
    diff = sharpe(s[idx]) - sharpe(h[idx])
    sr_s, sr_h = float(sharpe(s)), float(sharpe(h))
    p_sr = (1 + int((diff - diff.mean() >= sr_s - sr_h).sum())) / (1 + n_boot)   # centred null
    level = "too_early"
    if nw["n_eff"] >= min_eff:
        if a["p_pos"] < 0.01 and a_lo > 0:
            level = "strong"
        elif a["p_pos"] < 0.05 and a_lo > 0:
            level = "evidence"
        elif a["p_pos"] < 0.20:
            level = "hint"
        else:
            level = "none"
    return {**base, "level": level, "label": LEVELS[level], "ann": ann, "n_eff": nw["n_eff"],
            "excess_ann": nw["mean"] * ann, "excess_t": nw["t"], "excess_p": nw["p_pos"],
            "excess_ci_ann": (lo * ann, hi * ann),
            "beta": beta, "alpha_ann": a["mean"] * ann, "alpha_t": a["t"], "alpha_p": a["p_pos"],
            "alpha_ci_ann": (a_lo * ann, a_hi * ann),
            "sharpe": sr_s, "sharpe_hold": sr_h, "sharpe_diff": sr_s - sr_h,
            "sharpe_diff_ci": (float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))),
            "sharpe_diff_p": float(p_sr),
            "need": days_to_significance(a["t"], nw["n_eff"], n)}


# ------------------------------------------------------------------ selection bias
def deflated_sharpe(returns, n_trials: int, trial_sr_sd_ann: float | None = None,
                    ann: int | None = None, lab_ann: int = 252) -> dict:
    """Bailey & Lopez de Prado's deflated Sharpe. `returns`: the chosen strategy's periodic
    returns. `n_trials`: how many genomes were tried to find it. `trial_sr_sd_ann`: the spread
    (st. dev., annualised with `lab_ann`) of those genomes' Sharpe ratios; default 0.5.
    Returns the probability that the true Sharpe is above zero AFTER discounting the best-of-N
    selection (`dsr`), next to the undiscounted one (`psr`) and the Sharpe a pure-luck winner of
    N trials would show (`sr0_ann`)."""
    r = _finite(returns)
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return {"n": n, "dsr": float("nan"), "psr": float("nan")}
    ann = ann or 252
    sr = float(r.mean() / r.std(ddof=1))
    z = (r - r.mean()) / r.std(ddof=1)
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    sd_trials = (trial_sr_sd_ann if trial_sr_sd_ann is not None else 0.5) / math.sqrt(lab_ann)
    N = max(1, int(n_trials))
    sr0 = 0.0 if N == 1 else sd_trials * ((1 - EULER) * _N.inv_cdf(1 - 1.0 / N)
                                          + EULER * _N.inv_cdf(1 - 1.0 / (N * math.e)))
    psr = _N.cdf(sr * math.sqrt(n - 1) / denom)
    dsr = _N.cdf((sr - sr0) * math.sqrt(n - 1) / denom)
    return {"n": n, "n_trials": N, "sr_ann": sr * math.sqrt(ann), "sr0_ann": sr0 * math.sqrt(ann),
            "psr": float(psr), "dsr": float(dsr)}


# ------------------------------------------------------------------ how long would it take
def days_to_significance(t_obs: float, n_eff: float, n_now: int, target_t: float = 2.0):
    """If the effect seen so far is real and stays the same size, how many more sessions until
    t reaches `target_t` (t grows with sqrt of the sample). None if the effect is not positive.
    Treat as optimistic: winners' effects shrink."""
    if t_obs is None or not np.isfinite(t_obs) or t_obs <= 0 or not n_now:
        return None
    if t_obs >= target_t:
        return 0
    return int(math.ceil(n_now * ((target_t / t_obs) ** 2 - 1)))


def min_detectable(sd: float, n_eff: float, power_z: float = 0.84, alpha_z: float = 1.96):
    """Smallest true mean the data could reliably (80% power, 5% two-sided) have detected."""
    if not np.isfinite(sd) or not n_eff or n_eff < 1:
        return None
    return float((alpha_z + power_z) * sd / math.sqrt(n_eff))


def plain_summary(rep: dict) -> str:
    """One honest sentence for the dashboard / pipeline log."""
    if not rep:
        return "No evidence report yet."
    lvl = rep.get("level", "too_early")
    if "edge" in rep:
        e = rep["edge"]
        pp = e.get("p_perm")
        txt = (f"{rep['n_dates']} sessions (~{rep['n_eff']:.0f} independent): {LEVELS[lvl].lower()}")
        if pp == pp and pp is not None:
            txt += f" — random timing matches or beats the model {pp:.0%} of the time"
        return txt + "."
    if "alpha_ann" in rep:
        return (f"{rep['n']} sessions: {LEVELS[lvl].lower()} — alpha vs holding "
                f"{rep['alpha_ann']:+.1%}/yr (t={rep['alpha_t']:.1f}).")
    return LEVELS[lvl] + "."


# ------------------------------------------------------------------ ready-made reports
def live_evidence(journal_df, closes: pd.DataFrame, horizon: int, session_cfg=None, since=None,
                  long_only: bool = True, **kw) -> dict:
    """The live out-of-sample record: every after-close signal the pipeline journaled, scored on
    the prices that followed. These signals were produced before the returns existed, so no
    selection or leakage argument applies — this is the evidence that matters. `since` limits it
    to signals from that date on (e.g. since the sitting champion was promoted)."""
    from . import forwardtest
    ev = forwardtest.signal_events(journal_df, ("signals",))
    empty = {"n_dates": 0, "level": "too_early", "label": LEVELS["too_early"], "n_eff": 0.0}
    if ev is None or ev.empty:
        return empty
    rep = forwardtest.report(ev, closes, horizon=horizon, session_cfg=session_cfg)
    detail = rep.get("detail")
    if detail is None or len(detail) == 0 or "fwd_ret" not in detail:
        return empty
    if since is not None:
        detail = detail[detail["date"] >= pd.Timestamp(since)]
    out = edge_report(detail, horizon, long_only, **kw)
    if len(detail):
        out["first"], out["last"] = str(detail["date"].min())[:10], str(detail["date"].max())[:10]
    return out


def backtest_evidence(signals: pd.DataFrame, prices: pd.DataFrame, strat_ret: pd.Series,
                      hold_ret: pd.Series, horizon: int, n_trials: int | None = None,
                      trial_sr_sd: float | None = None, long_only: bool = True, **kw) -> dict:
    """The walk-forward backtest record, with the discount a backtest needs. The genome was
    picked because it did well on exactly this window, so: (1) its edge over holding is tested
    after beta (alpha), (2) if the lab's trial count is known, the excess return's Sharpe is
    deflated for the best-of-N selection, and the headline can't be better than 'a hint' unless
    it survives that (deflated Sharpe >= 95%). Only live results are free of this problem."""
    out = {"edge": edge_report(signal_detail(signals, prices, horizon), horizon, long_only, **kw),
           "vs_hold": excess_report(strat_ret, hold_ret, **{k: v for k, v in kw.items()
                                                            if k in ("n_boot", "seed", "min_eff")})}
    level = out["vs_hold"]["level"]
    if n_trials and n_trials > 1:
        df = pd.concat({"s": strat_ret, "h": hold_ret}, axis=1).dropna()
        d = deflated_sharpe((df["s"] - df["h"]).to_numpy(), n_trials, trial_sr_sd,
                            ann=periods_per_year(df.index))
        out["deflated"] = d
        if level in ("evidence", "strong") and not (d.get("dsr", 0.0) >= 0.95):
            level = "hint"
            out["downgraded"] = "excess return does not survive the best-of-N selection discount"
    out["level"], out["label"] = level, LEVELS[level]
    return out


def jsonable(o):
    """numpy/pandas -> plain JSON types (NaN/inf -> None) for data/evidence.json and journal events."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)[:10]
    return o


def brief_live(rep: dict) -> dict:
    """The few numbers worth journaling from a live_evidence() report."""
    if not rep:
        return {}
    e, c = rep.get("edge") or {}, rep.get("ic_cs") or {}
    ci = e.get("ci") or (None, None)
    return jsonable({
        "level": rep.get("level"), "label": rep.get("label"), "n_dates": rep.get("n_dates"),
        "n_eff": rep.get("n_eff"), "first": rep.get("first"), "last": rep.get("last"),
        "edge_mean": e.get("mean"), "edge_ci_lo": ci[0], "edge_ci_hi": ci[1],
        "p_perm": e.get("p_perm"), "p_nw": e.get("p_nw"),
        "random_timing_mean": e.get("random_timing_mean"),
        "ic_cs": c.get("mean"), "ic_cs_t": c.get("t"), "ic_pooled": rep.get("ic_pooled"),
        "more_sessions_needed": rep.get("need"), "detectable_edge": rep.get("detectable")})


def brief_backtest(be: dict) -> dict:
    """The few numbers worth journaling from a backtest_evidence() report."""
    if not be:
        return {}
    v, e, d = be.get("vs_hold") or {}, be.get("edge") or {}, be.get("deflated") or {}
    return jsonable({
        "level": be.get("level"), "label": be.get("label"), "downgraded": be.get("downgraded"),
        "n": v.get("n"), "n_eff": v.get("n_eff"), "alpha_ann": v.get("alpha_ann"),
        "alpha_t": v.get("alpha_t"), "alpha_p": v.get("alpha_p"), "beta": v.get("beta"),
        "excess_ann": v.get("excess_ann"), "sharpe_diff": v.get("sharpe_diff"),
        "sharpe_diff_ci": v.get("sharpe_diff_ci"),
        "p_perm": (e.get("edge") or {}).get("p_perm"),
        "n_trials": d.get("n_trials"), "dsr": d.get("dsr"), "sr0_ann": d.get("sr0_ann")})


def lab_trials(events, default_per_run: int = 0) -> tuple[int, float | None]:
    """How many genomes the lab has tried in total, and how far apart their Sharpes typically
    were, from the journal's evolution events (dicts). Runs logged before the lab began recording
    its counts are assumed to have tried the median of the logged ones (or `default_per_run`)."""
    events = list(events or [])
    ns = [int(e["n_evaluated"]) for e in events if e.get("n_evaluated")]
    sds = [float(e["trial_sharpe_sd"]) for e in events if e.get("trial_sharpe_sd")]
    typical = int(np.median(ns)) if ns else int(default_per_run)
    total = sum(ns) + typical * (len(events) - len(ns))
    return max(int(total), 1), (float(np.median(sds)) if sds else None)


# ------------------------------------------------------------------ prices in the journal
def closes_from_journal(journal_df) -> pd.DataFrame:
    """Daily closes the pipeline logged as "closes" events, oldest first (later events win).
    With signals and closes both in journal.db, the whole live record can be re-tested from the
    repo alone, with no market-data keys."""
    import json
    rows = {}
    if journal_df is None or journal_df.empty:
        return pd.DataFrame()
    for _, r in journal_df.sort_values("ts").iterrows():
        if r["kind"] != "closes":
            continue
        try:
            for d, px in (json.loads(r["payload"]).get("closes") or {}).items():
                rows.setdefault(d, {}).update({k: v for k, v in px.items() if v is not None})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.to_datetime(out.index)
    return out.sort_index().astype(float)
