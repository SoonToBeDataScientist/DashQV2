"""The live evidence path end to end on a synthetic journal: journaled signals -> verdict ->
data/evidence.json + a journal event, for a real edge and for noise."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_evidence import world  # noqa: E402

from src import evidence as ev, journal, pipeline as P  # noqa: E402


def make_journal(path, sig, hour=21, minute=30):
    for d, row in sig.iterrows():
        ts = (pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=minute)).tz_localize("UTC").isoformat()
        with journal._conn(path) as c:
            c.execute("INSERT INTO events VALUES (?,?,?)", (ts, "signals",
                                                            json.dumps({k: float(v) for k, v in row.items()})))


def run(ic, seed):
    sig, px, h = world(T=320, ic=ic, persist=0.5, drift=0.0006, seed=seed)
    tmp = tempfile.mkdtemp()
    jp = os.path.join(tmp, "journal.db")
    make_journal(jp, sig)
    s = SimpleNamespace(data_dir=tmp)
    brief = P.record_live_evidence(s, {}, jp, px, h, allow_short=False)
    return tmp, jp, brief


def test_edge_and_noise_end_to_end():
    tmp, jp, edge = run(0.30, 1)
    assert edge and edge["level"] in ("evidence", "strong"), edge
    saved = json.load(open(os.path.join(tmp, "evidence.json")))
    assert saved["live"]["level"] == edge["level"] and saved["live"]["asof"]
    jdf = journal.read(jp, kind="evidence")
    assert len(jdf) == 1 and json.loads(jdf["payload"].iloc[0])["p_perm"] == edge["p_perm"]
    json.dumps(saved)                                           # plain JSON, no numpy/NaN leftovers

    _, _, noise = run(0.0, 2)
    assert noise["level"] in ("none", "hint"), noise


def test_closes_round_trip_rebuilds_the_verdict_from_the_journal_alone():
    sig, px, h = world(T=320, ic=0.30, persist=0.5, drift=0.0006, seed=8)
    tmp = tempfile.mkdtemp()
    jp = os.path.join(tmp, "journal.db")
    make_journal(jp, sig)
    bars = {c: pd.DataFrame({"close": px[c]}) for c in px}
    n1 = P.log_closes(jp, bars)
    assert n1 == 300 and P.log_closes(jp, bars) == 0                 # second call: nothing new
    bars2 = {c: pd.DataFrame({"close": px[c]}) for c in px}
    extra = pd.Series([200.0], index=[px.index[-1] + pd.offsets.BDay(1)])
    bars2["S00"] = pd.DataFrame({"close": pd.concat([px["S00"], extra])})
    assert P.log_closes(jp, bars2) == 1                              # only the new day is added
    back = ev.closes_from_journal(journal.read(jp, limit=10000))
    assert back.shape[1] == px.shape[1] and abs(back["S01"].loc[px.index[-1]] - px["S01"].iloc[-1]) < 1e-3
    live = ev.live_evidence(journal.read(jp, limit=10000), back, h, session_cfg={})
    assert live["level"] in ("evidence", "strong", "hint"), live["level"]


def test_lab_writer_keeps_live_verdict():
    tmp, jp, _ = run(0.3, 3)
    P._write_evidence(SimpleNamespace(data_dir=tmp), "lab", {"level": "hint", "asof": "2026-11-01"})
    saved = json.load(open(os.path.join(tmp, "evidence.json")))
    assert set(saved) == {"live", "lab"} and saved["lab"]["level"] == "hint"


def test_lab_trials_counts_every_run():
    evs = [{"n_evaluated": 12, "trial_sharpe_sd": 0.4}, {"n_evaluated": 18}, {}, {}]
    n, sd = ev.lab_trials(evs, default_per_run=10)
    assert n == 12 + 18 + 15 + 15 and abs(sd - 0.4) < 1e-9          # unlogged runs get the median (15)
    assert ev.lab_trials([], 7) == (1, None)


def test_backtest_evidence_discounts_selection():
    sig, px, h = world(T=500, ic=0.12, persist=0.5, drift=0.0006, seed=4)
    rng = np.random.default_rng(0)
    hold = pd.Series(rng.normal(0.0005, 0.01, 500), index=sig.index)
    strat = 0.5 * hold + 0.0004 + rng.normal(0, 0.003, 500)          # real but modest alpha
    free = ev.backtest_evidence(sig, px, strat, hold, h, n_boot=300, max_shifts=300)
    picked = ev.backtest_evidence(sig, px, strat, hold, h, n_trials=800, trial_sr_sd=1.0,
                                  n_boot=300, max_shifts=300)
    assert free["vs_hold"]["alpha_ann"] > 0 and "deflated" not in free
    ex = (strat - hold).to_numpy()
    undiscounted = ev.deflated_sharpe(ex, 1)["dsr"]
    d = picked["deflated"]
    assert d["n_trials"] == 800 and d["dsr"] < undiscounted and d["sr0_ann"] > 0.5
    # a headline of "evidence" must survive the discount, otherwise it is capped at "hint"
    if picked["level"] in ("evidence", "strong"):
        assert d["dsr"] >= 0.95
    else:
        assert picked["level"] in ("hint", "none", "too_early")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
    print("EVIDENCE PIPELINE TESTS PASSED")
