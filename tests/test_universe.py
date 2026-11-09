"""The 30-symbol universe: the pieces that had to change for it, and that nothing changed for the
original six."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from src import backtest as bt, models, pipeline as P  # noqa: E402


def _xy(n, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.standard_normal((n, 6)), columns=list("abcdef"))
    y = pd.Series(0.3 * X["a"] + r.standard_normal(n))
    return X, y


def test_random_forest_unchanged_for_small_universes_and_capped_for_big_ones():
    from sklearn.ensemble import RandomForestRegressor
    g = models.Genome(name="rf", model_type="rf", max_depth=4)
    X, y = _xy(3000)                                           # ~6 assets x a 504-day window
    m = models.SignalModel(g).fit(X, y)
    assert m.model.max_samples is None
    ref = RandomForestRegressor(n_estimators=200, max_depth=g.max_depth + 2, min_samples_leaf=10,
                                n_jobs=-1, random_state=7)
    Xc = X.fillna(m.med_).fillna(0.0).clip(-8, 8)
    ref.fit(Xc, y.clip(y.quantile(0.02), y.quantile(0.98)))
    assert np.allclose(ref.predict(Xc), m.model.predict(Xc))      # identical fit below the cap
    Xb, yb = _xy(15000, 1)                                     # ~30 assets
    mb = models.SignalModel(g).fit(Xb, yb)
    assert abs(mb.model.max_samples - models.RF_MAX_ROWS / 15000) < 1e-12


def test_vol_target_cap_is_configurable_and_defaults_to_the_old_value():
    idx = pd.bdate_range("2025-01-01", periods=200)
    r = np.random.default_rng(1)
    px = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.004, (200, 3)), axis=0)), index=idx, columns=list("ABC"))
    sig = pd.DataFrame(0.05, index=idx, columns=px.columns)
    try:
        assert bt.configure_vol_target(3.0) == 3.0
        w3 = bt.target_weights(sig, px, 2.0, "equal+vt12", allow_short=False)
        assert np.isclose(w3.iloc[-1].abs().sum() / (0.05 * 2.0), 3.0)               # cap binds at the old 3x
        bt.configure_vol_target(5)
        w5 = bt.target_weights(sig, px, 2.0, "equal+vt12", allow_short=False)
        assert np.isclose(w5.iloc[-1].abs().sum() / (0.05 * 2.0), 5.0)
        assert bt.configure_vol_target("junk") == 3.0 and bt.configure_vol_target(0.2) == 3.0
    finally:
        bt.configure_vol_target(3.0)


def test_sentiment_symbols_pin():
    syms = ["AAPL", "QQQ", "BTC/USD", "TLT"]
    assert P._sentiment_symbols({}, syms) == syms
    assert P._sentiment_symbols({"sentiment_symbols": ["AAPL", "BTC/USD", "ZZZ"]}, syms) == ["AAPL", "BTC/USD"]


def test_shipped_configs_are_consistent():
    for f in ("universe.json", "universe.example.json"):
        c = json.load(open(os.path.join(ROOT, f)))
        stocks, cryptos = c["stocks"], c["cryptos"]
        allsyms = stocks + cryptos
        assert len(set(allsyms)) == len(allsyms), "duplicate symbol"
        assert all("/" not in s and s == s.upper() for s in stocks) and all(s.endswith("/USD") for s in cryptos)
        assert set(c.get("sentiment_symbols", allsyms)) <= set(allsyms)
        assert set(c.get("snapshot_symbols", allsyms)) <= set(allsyms)
        assert float(c.get("vt_max_scale", 3.0)) >= 1.0
        assert len(allsyms) >= 6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
    print("UNIVERSE TESTS PASSED")
