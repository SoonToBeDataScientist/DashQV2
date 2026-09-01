from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import backtest as bt
from . import datafeed, exodata, journal, models
from .config import load_settings
from .features import FEATURE_GROUPS, build_panel
from .models import Genome


def _clean(d):
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def load_champion(s) -> Genome:
    p = os.path.join(s.model_dir, "champion_genome.json")
    if os.path.exists(p):
        return Genome(**json.load(open(p)))
    b = models.load_bundle(os.path.join(s.model_dir, "champion.joblib"))
    return b["genome"] if b else Genome()


def save_champion(s, genome, model=None):
    import shutil
    os.makedirs(s.model_dir, exist_ok=True)
    cur = os.path.join(s.model_dir, "champion_genome.json")
    if os.path.exists(cur):                                    # keep previous champion for rollback
        shutil.copy(cur, os.path.join(s.model_dir, "champion_genome.prev.json"))
        bj = os.path.join(s.model_dir, "champion.joblib")
        if os.path.exists(bj):
            shutil.copy(bj, os.path.join(s.model_dir, "champion.prev.joblib"))
    json.dump(asdict(genome), open(cur, "w"), indent=2)
    if model is not None:
        models.save_bundle(os.path.join(s.model_dir, "champion.joblib"), genome, model)


def compute_live_ic(jpath: str, bars: dict, horizon: int):
    """Spearman IC between journaled signals and realized forward returns. Cheap —
    no retraining, just correlation on data that's already local (journal + closes)."""
    try:
        from . import forwardtest
        jdf = journal.read(jpath, limit=10000)
        closes = pd.DataFrame({k: v["close"] for k, v in bars.items()})
        rep = forwardtest.report(forwardtest.signal_events(jdf), closes, horizon=horizon)
        return rep.get("ic"), rep.get("n", 0)
    except Exception:
        return None, 0


def evolved_today(jpath: str) -> bool:
    """Cooldown check: has the Strategy Lab already run today (any trigger)?"""
    df = journal.read(jpath, kind="evolution", limit=20)
    if df.empty:
        return False
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    return bool((pd.to_datetime(df["ts"]).dt.tz_localize(None) >= today).any())


def _search_and_maybe_promote(s, cfg, panel, genome, jpath, trigger, live_ic=None, n_obs=0):
    """Run the Strategy Lab (search + champion/challenger comparison) and promote if the
    challenger clears promote_margin. Shared by the nightly run() and the intraday check —
    this never touches trading, only the champion files."""
    evo = cfg.get("evolution", {})
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=365 * evo.get("eval_years", 2))
    pe = panel[panel["date"] >= cutoff]
    cost = cfg.get("fee_bps", 2) + cfg.get("slippage_bps", 5)

    if evo.get("method", "evolution") == "optuna":
        from . import optimize
        storage = f"sqlite:///{os.path.join(s.data_dir, 'optuna.db')}"   # warm-starts weekly
        rows, _, _ = optimize.optimize(pe, n_trials=evo.get("trials", 25),
                                       cost_bps=cost, storage=storage)
    else:
        rows, _ = models.evolve(pe, pop_size=evo.get("pop_size", 6),
                                generations=evo.get("generations", 2), cost_bps=cost)

    prices = pe.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    champ_sig, _ = models.walk_forward_signals(pe, genome)
    cm = bt.run_backtest(prices, champ_sig, cfg.get("fee_bps", 2) / 2,
                         cfg.get("slippage_bps", 5) / 2, genome.max_leverage).metrics
    champ_score = cm["sharpe"] - 0.5 * abs(cm["max_drawdown"])
    challenger = rows[0]
    event = {"trigger": trigger, "champion": genome.name,
             "champion_score": round(champ_score, 3),
             "champion_live_ic": (round(live_ic, 3) if live_ic is not None else None),
             "champion_live_obs": n_obs,
             "challenger": challenger["name"], "challenger_score": challenger["score"]}
    if challenger["score"] > champ_score + evo.get("promote_margin", 0.05):
        g = challenger["genome"]
        save_champion(s, g, models.train_final_model(panel, g))
        event["promoted"] = g.name
    else:
        event["promoted"] = None
    journal.log(jpath, "evolution", event)
    return event


def check_intraday(config_path="universe.json") -> dict | None:
    """Lightweight IC health check — safe to call every ~30 min during market hours.
    Step 1 is cheap (just closes + the journal, no retraining). It only escalates to the
    heavier full-history fetch + Strategy Lab run if the live IC has genuinely degraded,
    AND the lab hasn't already run today (cooldown) — so this reacts to real deterioration
    without noise-chasing on the same daily-bar data over and over."""
    s = load_settings()
    cfg = json.load(open(config_path))
    evo = cfg.get("evolution", {})
    if not (evo.get("enabled") and evo.get("intraday_reaction", True)):
        return None
    jpath = os.path.join(s.data_dir, "journal.db")
    if evolved_today(jpath):
        return None

    stocks, cryptos = cfg.get("stocks", []), cfg.get("cryptos", [])
    symbols = stocks + cryptos
    asset_map = {**{x: "stock" for x in stocks}, **{x: "crypto" for x in cryptos}}
    genome = load_champion(s)

    short_start = dt.datetime.utcnow() - dt.timedelta(days=400)   # plenty for IC on recent journal entries
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, short_start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 20}
    if not bars:
        return None
    live_ic, n_obs = compute_live_ic(jpath, bars, genome.horizon)
    panic = (live_ic is not None and n_obs >= evo.get("ic_min_obs", 60)
             and live_ic < evo.get("ic_panic_threshold", -0.02))
    if not panic:
        return None

    # confirmed panic -> fetch full history and run the (rare, heavier) Strategy Lab
    start = dt.datetime.utcnow() - dt.timedelta(days=365 * cfg.get("years", 4))
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 120}
    macro = datafeed.get_macro(start, s)
    sent = datafeed.get_news_sentiment(symbols, s, backend=cfg.get("sentiment_backend", "auto"))
    exo = exodata.get_exo(start, os.path.join(s.data_dir, "snapshots.csv"))   # read-only snapshots, no new scraping
    full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
    panel = build_panel(bars, macro, sent, full, exo)
    return _search_and_maybe_promote(s, cfg, panel, genome, jpath, "ic_panic", live_ic, n_obs)


def run(config_path="universe.json", force_evolve=False, no_trade=False) -> dict:
    s = load_settings()
    cfg = json.load(open(config_path))
    stocks, cryptos = cfg.get("stocks", []), cfg.get("cryptos", [])
    symbols = stocks + cryptos
    asset_map = {**{x: "stock" for x in stocks}, **{x: "crypto" for x in cryptos}}
    start = dt.datetime.utcnow() - dt.timedelta(days=365 * cfg.get("years", 4))
    jpath = os.path.join(s.data_dir, "journal.db")
    out = {"ts": pd.Timestamp.now("UTC").isoformat(), "symbols": symbols}

    # 1. data
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 120}
    macro = datafeed.get_macro(start, s)
    sent = datafeed.get_news_sentiment(symbols, s, backend=cfg.get("sentiment_backend", "auto"))
    snaps = exodata.collect_snapshots(symbols, asset_map, os.path.join(s.data_dir, "snapshots.csv"))
    exo = {"options_market": exodata.options_market_history(start),
           "onchain": exodata.onchain_history(), "snapshots": snaps}

    # 2. champion model + latest signals
    genome = load_champion(s)
    full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
    panel = build_panel(bars, macro, sent, full, exo)
    model = models.train_final_model(panel, genome)
    latest = models.recent_signals(panel, genome, model).iloc[-1]
    out["genome"], out["signals"] = genome.name, _clean({k: round(float(v), 4)
                                                         for k, v in latest.items()})
    if not macro.empty:
        m = macro.iloc[-1]
        out["regime"] = _clean({"vix": round(float(m.get("vix_lvl", np.nan)), 2),
                                "vix_z": round(float(m.get("vix_z", np.nan)), 2),
                                "spy_ret21": round(float(m.get("spy_ret21", np.nan)), 4)})
    json.dump(out, open(os.path.join(s.data_dir, "latest_signals.json"), "w"),
              indent=2, default=str)
    journal.log(jpath, "signals", out["signals"])

    # 3. paper-trade rebalance (Alpaca queues stock orders placed after close; crypto fills now)
    if cfg.get("trade", True) and not no_trade and datafeed.has_alpaca(s):
        tc = datafeed.trading_client(s)
        try:
            out["market_open"] = bool(tc.get_clock().is_open)
        except Exception:
            out["market_open"] = None
        w = latest.copy()
        if not cfg.get("allow_short", False):
            w = w.clip(lower=0)
        for sym in cryptos:                                   # Alpaca: no crypto shorts
            if sym in w:
                w[sym] = max(w[sym], 0.0)
        targets = (w * (genome.max_leverage / max(1, len(w))) * float(tc.get_account().equity)).to_dict()
        results = datafeed.rebalance(tc, targets, asset_map)
        journal.log(jpath, "orders", results)
        out["orders"] = results

    # 4. autonomous Strategy Lab: scheduled + adaptive (live-IC-triggered)
    evo = cfg.get("evolution", {})
    lab_due = bool(evo.get("enabled")) and dt.datetime.utcnow().weekday() == int(evo.get("weekday", 5))
    live_ic, n_obs = compute_live_ic(jpath, bars, genome.horizon)
    ic_panic = (not evolved_today(jpath) and live_ic is not None
                and n_obs >= evo.get("ic_min_obs", 60)
                and live_ic < evo.get("ic_panic_threshold", -0.02))

    if force_evolve or lab_due or (evo.get("enabled") and ic_panic):
        trigger = "manual" if force_evolve else ("ic_panic" if ic_panic and not lab_due else "scheduled")
        event = _search_and_maybe_promote(s, cfg, panel, genome, jpath, trigger, live_ic, n_obs)
        out["evolution"] = event
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe.json")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--no-trade", action="store_true")
    ap.add_argument("--check-intraday", action="store_true",
                    help="run only the cheap IC health check (+ Strategy Lab if it panics), "
                         "no data refresh/trading — for a tighter-cadence cron than the daily run")
    a = ap.parse_args()
    if a.check_intraday:
        event = check_intraday(a.config)
        print(json.dumps({"status": "ok", "evolution": event}, default=str))
    else:
        print(json.dumps({"status": "ok", **run(a.config, a.evolve, a.no_trade)}, default=str))


if __name__ == "__main__":
    main()
