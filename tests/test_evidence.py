"""Calibration tests for src/evidence.py on synthetic worlds with a KNOWN answer.

Run:  python -m pytest tests/test_evidence.py -q     (or)     python tests/test_evidence.py
The point of these tests is that the edge test must (a) find an edge that is really there,
(b) NOT find one that isn't — at roughly its nominal 5% rate — even when the market trends up,
signals are persistent and forward returns overlap, which are exactly the conditions that make
naive tests cheer for nothing.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import evidence as ev  # noqa: E402

SYMS = [f"S{i:02d}" for i in range(40)]


def world(T=250, K=6, h=3, drift=0.0006, ic=0.0, persist=0.0, seed=0, vol=0.01):
    """Daily returns with a market factor and drift. Signal = ic * (true next-h-day return,
    standardised) + noise, optionally smoothed to be persistent. ic=0 -> pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=T + h + 5)
    m = rng.normal(0, 0.008, len(idx))
    r = drift + m[:, None] * rng.uniform(0.6, 1.2, K) + rng.normal(0, vol, (len(idx), K))
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.log1p(r), axis=0)), index=idx, columns=SYMS[:K])
    fwd = px.shift(-h) / px - 1
    z = (fwd - fwd.mean()) / fwd.std()
    noise = rng.normal(0, 1, z.shape)
    raw = ic * z.fillna(0).to_numpy() + np.sqrt(max(1e-9, 1 - ic ** 2)) * noise
    sig = pd.DataFrame(raw, index=idx, columns=px.columns)
    if persist:
        sig = sig.ewm(alpha=1 - persist).mean()
        sig = sig / sig.std()
    sig = np.tanh(0.5 * sig).iloc[:T]
    return sig, px, h


def test_nw_matches_plain_t_for_iid():
    x = np.random.default_rng(1).normal(0.1, 1, 2000)
    r = ev.nw_mean_test(x, lags=0)
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(r["t"] - plain) / plain < 0.02
    assert r["n_eff"] > 0.9 * len(x)


def test_nw_deflates_overlapping_returns():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 3000)
    h = 5
    x = pd.Series(base).rolling(h).sum().dropna().to_numpy()          # overlapping h-day sums
    naive = ev.nw_mean_test(x, lags=0)
    robust = ev.nw_mean_test(x, lags=h - 1)
    assert robust["n_eff"] < 0.4 * naive["n_eff"]                    # ~n/h
    assert abs(robust["t"]) < abs(naive["t"])


def test_signal_detail_matches_forwardtest_report():
    from src import forwardtest
    rng = np.random.default_rng(3)
    cal = pd.date_range("2026-01-01", periods=120, freq="D")
    closes = pd.DataFrame(index=cal)
    closes["STK"] = np.where(cal.dayofweek < 5, 100 * np.exp(np.cumsum(rng.normal(0, .01, len(cal)))), np.nan)
    closes["BTC"] = 100 * np.exp(np.cumsum(rng.normal(0, .02, len(cal))))
    dates = cal[10:90]
    sig = pd.DataFrame({"STK": rng.normal(0, 1, len(dates)), "BTC": rng.normal(0, 1, len(dates))}, index=dates)
    sig.loc[sig.index.dayofweek >= 5, "STK"] = np.nan                 # no fresh stock signal on weekends
    got = ev.signal_detail(sig, closes, horizon=3)
    rows = [{"ts": d, "symbol": s, "signal": v} for d in sig.index for s, v in sig.loc[d].items() if v == v]
    old = forwardtest.report(pd.DataFrame(rows), closes, horizon=3)["detail"]
    m = got.merge(old, on=["date", "symbol"], suffixes=("", "_old"))
    assert len(m) == len(old) == len(got)
    assert np.allclose(m["fwd_ret"], m["fwd_ret_old"], atol=1e-12)
    assert np.allclose(m["signal"], m["signal_old"], atol=1e-12)


def test_planted_edge_is_found():
    hits = 0
    for seed in range(10):
        sig, px, h = world(T=400, ic=0.25, persist=0.5, seed=seed)
        rep = ev.edge_report(ev.signal_detail(sig, px, h), h, n_boot=300, max_shifts=300, seed=seed)
        hits += rep["level"] in ("evidence", "strong")
    assert hits >= 8, hits


def test_noise_is_rejected_at_nominal_rate_despite_drift_and_persistence():
    """Persistent noise signals in a rising market, long-only: the traded book makes money on drift
    alone. The permutation verdict must still fire ~5% of the time, not ~50%."""
    n, fired, naive = 160, 0, 0
    for seed in range(n):
        sig, px, h = world(T=250, ic=0.0, persist=0.9, drift=0.0012, seed=1000 + seed)
        d = ev.signal_detail(sig, px, h)
        rep = ev.edge_report(d, h, n_boot=100, max_shifts=200, seed=seed)
        fired += rep["edge"]["p_perm"] < 0.05
        naive += rep["edge"]["p_nw"] < 0.05          # what a plain "book made money" t-test would say
    rate = fired / n
    assert rate < 0.10, f"false-positive rate {rate:.2%}"
    assert naive > fired + 10, "this world should fool the naive test far more than the permutation"
    print(f"  noise: permutation fires {rate:.1%}, naive t-test fires {naive / n:.1%}")


def test_always_long_in_a_bull_market_is_not_an_edge():
    sig, px, h = world(T=300, ic=0.0, drift=0.002, seed=5)
    sig[:] = 0.5
    rep = ev.edge_report(ev.signal_detail(sig, px, h), h, n_boot=200, max_shifts=200)
    assert rep["edge"]["p_nw"] < 0.05                  # looks great to a naive test...
    assert rep["edge"]["p_perm"] > 0.9 and rep["level"] == "none"   # ...but is just exposure


def test_too_early_with_few_sessions():
    sig, px, h = world(T=40, ic=0.4, seed=7)
    rep = ev.edge_report(ev.signal_detail(sig, px, h), h, n_boot=100, max_shifts=100)
    assert rep["level"] == "too_early"


def test_excess_report_alpha_beta():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=900)
    hold = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    half = 0.5 * hold + rng.normal(0, 0.002, len(idx))               # half the exposure, no skill
    r = ev.excess_report(half, hold, n_boot=400)
    assert abs(r["beta"] - 0.5) < 0.05 and r["level"] in ("none", "hint"), r["level"]
    skilled = 0.5 * hold + 0.0008 + rng.normal(0, 0.002, len(idx))   # +20%/yr alpha at half the risk
    r2 = ev.excess_report(skilled, hold, n_boot=400)
    assert r2["alpha_ann"] > 0.1 and r2["level"] in ("evidence", "strong"), (r2["alpha_ann"], r2["level"])
    assert ev.excess_report(hold.iloc[:10], hold.iloc[:10])["level"] == "too_early"


def test_excess_false_positive_rate():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2022-01-03", periods=500)
    fired = 0
    for _ in range(150):
        hold = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
        strat = 0.6 * hold + rng.normal(0, 0.004, len(idx))
        fired += ev.excess_report(strat, hold, n_boot=60)["alpha_p"] < 0.05
    assert fired / 150 < 0.11, fired / 150


def test_deflated_sharpe():
    rng = np.random.default_rng(31)
    r = rng.normal(0.0006, 0.01, 750)                                   # ~ Sharpe 0.95 over 3 years
    one = ev.deflated_sharpe(r, 1)
    assert abs(one["dsr"] - one["psr"]) < 1e-9
    many = ev.deflated_sharpe(r, 500, trial_sr_sd_ann=0.8)
    assert many["dsr"] < one["dsr"] and many["sr0_ann"] > 1.0
    strong = ev.deflated_sharpe(rng.normal(0.003, 0.01, 750), 500, trial_sr_sd_ann=0.8)
    assert strong["dsr"] > 0.99
    assert np.isnan(ev.deflated_sharpe(r[:10], 5)["dsr"])


def test_power_helpers():
    assert ev.days_to_significance(1.0, 30, 40) == 120      # 40 * (4 - 1)
    assert ev.days_to_significance(2.5, 30, 40) == 0
    assert ev.days_to_significance(-0.5, 30, 40) is None
    assert ev.min_detectable(0.45, 100) == 2.8 * 0.45 / 10
    assert ev.periods_per_year(pd.bdate_range("2024-01-01", periods=30)) == 252
    assert ev.periods_per_year(pd.date_range("2024-01-01", periods=30)) == 365


if __name__ == "__main__":
    import time
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"OK  {name}  ({time.time() - t0:.1f}s)")
    print("ALL EVIDENCE TESTS PASSED")
