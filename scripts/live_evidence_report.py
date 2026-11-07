"""Re-test the live signal record from the repo alone (journal.db holds both the signals and the
closes they were judged on):

    python scripts/live_evidence_report.py [--journal data/journal.db] [--config universe.json]

Prints the headline edge test, then the checks that tell whether it is fragile: leave-one-asset-out,
crypto vs stocks, other forward horizons, first vs second half, and the same test applied to trivial
momentum rules and to always-long in the same window (if those "win" too, the window was just easy).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import evidence as ev, forwardtest, journal  # noqa: E402

warnings.filterwarnings("ignore")


def line(tag, r):
    e = r.get("edge")
    if not e:
        print(f"{tag:36s} n_dates={r['n_dates']:3d}  (too few sessions)")
        return
    print(f"{tag:36s} n_dates={r['n_dates']:3d} n_eff={r['n_eff']:5.1f} edge/session={e['mean']:+.3%} "
          f"95%CI=[{e['ci'][0]:+.3%},{e['ci'][1]:+.3%}] random-timing-matches={e['p_perm']:.1%} "
          f"IC(per date)={r['ic_cs']['mean']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="data/journal.db")
    ap.add_argument("--config", default="universe.json")
    ap.add_argument("--genome", default="models/champion_genome.json")
    a = ap.parse_args()
    j = journal.read(a.journal, limit=50000)
    closes = ev.closes_from_journal(j)
    if closes.empty:
        sys.exit("No 'closes' events in the journal yet — the daily pipeline logs them from its next run on.")
    cfg, H = json.load(open(a.config)), json.load(open(a.genome))["horizon"]
    long_only = not cfg.get("allow_short", False)
    D = forwardtest.report(forwardtest.signal_events(j, ("signals",)), closes, horizon=H, session_cfg=cfg)["detail"]
    if D is None or len(D) < 10:
        sys.exit("Not enough matured signals yet.")
    print(f"{D['date'].nunique()} matured sessions, {D['date'].min():%Y-%m-%d} -> {D['date'].max():%Y-%m-%d}, "
          f"horizon {H} sessions, {'long-only' if long_only else 'long/short'}\n")
    base = ev.edge_report(D, H, long_only, seed=1)
    line("HEADLINE", base)
    print("  ->", ev.plain_summary(base), "\n")

    print("-- is it one asset?")
    for s in sorted(D.symbol.unique()):
        line(f"without {s}", ev.edge_report(D[D.symbol != s], H, long_only, n_boot=500, seed=2))
    cr = [x for x in D.symbol.unique() if "/" in x]
    line("crypto only", ev.edge_report(D[D.symbol.isin(cr)], H, long_only, n_boot=500, seed=3))
    line("stocks only", ev.edge_report(D[~D.symbol.isin(cr)], H, long_only, n_boot=500, seed=3))

    print("\n-- other forward horizons (same signals)")
    sig = D.pivot_table(index="date", columns="symbol", values="signal")
    for h in (1, 2, 3, 5, 10):
        line(f"horizon {h}", ev.edge_report(ev.signal_detail(sig, closes, h), h, long_only, n_boot=500, seed=4))

    print("\n-- first vs second half")
    dts = np.sort(D["date"].unique())
    mid = dts[len(dts) // 2]
    line(f"before {pd.Timestamp(mid):%Y-%m-%d}", ev.edge_report(D[D.date < mid], H, long_only, n_boot=500, seed=5))
    line(f"from {pd.Timestamp(mid):%Y-%m-%d}", ev.edge_report(D[D.date >= mid], H, long_only, n_boot=500, seed=6))

    print("\n-- the same test on trivial rules in the same window (is the window just easy?)")
    dates = pd.DatetimeIndex(dts)
    for lb in (5, 20, 50):
        rule = pd.DataFrame({s: (closes[s].dropna() / closes[s].dropna().shift(lb) - 1 > 0).astype(float)
                             .reindex(dates, method="ffill") for s in closes})
        d = ev.signal_detail(rule, closes, H)
        line(f"long if {lb}-day return > 0", ev.edge_report(d[d.date.isin(dates)], H, True, n_boot=500, seed=7))
    hold = ev.signal_detail(pd.DataFrame(1.0, index=dates, columns=sorted(D.symbol.unique())), closes, H)
    line("always long", ev.edge_report(hold, H, True, n_boot=500, seed=8))

    print("\n-- by asset: forward return after a positive signal vs after none")
    for s in sorted(D.symbol.unique()):
        x = D[D.symbol == s]
        pos, neg = x[x.signal > 0], x[x.signal <= 0]
        print(f"  {s:9s} signal>0: {pos.fwd_ret.mean():+.2%} (n={len(pos):2d})   signal<=0: {neg.fwd_ret.mean():+.2%} (n={len(neg):2d})")


if __name__ == "__main__":
    main()
