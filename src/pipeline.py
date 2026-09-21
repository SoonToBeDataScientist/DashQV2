from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from . import backtest as bt
from . import datafeed, exodata, journal, models, session
from .config import load_settings
from .features import FEATURE_GROUPS, build_panel, feature_columns
from .models import Genome


def _clean(d):
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _universe(cfg):
    stocks, cryptos = cfg.get("stocks", []), cfg.get("cryptos", [])
    return stocks, cryptos, stocks + cryptos, {**{x: "stock" for x in stocks},
                                               **{x: "crypto" for x in cryptos}}


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


# ---------------------------------------------------------------- session bookkeeping
def _events(jpath: str, kind: str, limit=500) -> pd.DataFrame:
    df = journal.read(jpath, kind=kind, limit=limit)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


def _evolution_sessions(jpath: str, cfg) -> pd.Series:
    df = _events(jpath, "evolution", 200)
    return session.session_dates(cfg, df["ts"]) if not df.empty else pd.Series(dtype="datetime64[ns]")


def evolved_on(jpath: str, cfg, day) -> bool:
    """Cooldown: has the Strategy Lab already run for this session (any trigger)?"""
    return bool((_evolution_sessions(jpath, cfg) == day).any())


def lab_due(jpath: str, cfg, day) -> bool:
    """Weekly lab keyed on the SESSION's weekday, so a Saturday run that GitHub starts after
    midnight UTC still counts as Saturday — plus a catch-up if a whole scheduled run was dropped."""
    evo = cfg.get("evolution", {})
    past = _evolution_sessions(jpath, cfg)
    if not evo.get("enabled") or (past == day).any():
        return False
    if day.weekday() == int(evo.get("weekday", 5)):
        return True
    return not past.empty and (day - past.max()).days >= 8


def ran_for_session(jpath: str, cfg, day) -> bool:
    """Did an after-close run already log signals for this session? Lets a backup cron exit
    early — and a mid-session manual run doesn't count, so it can't block the real one."""
    df = _events(jpath, "signals", 50)
    if df.empty:
        return False
    same = session.session_dates(cfg, df["ts"]).values == np.datetime64(day)
    return bool((same & (df["ts"] >= session.close_utc(cfg, day)).values).any())


def compute_live_ic(jpath: str, bars: dict, horizon: int, cfg=None):
    """Spearman IC between journaled signals and realized forward returns. Cheap —
    no retraining, just correlation on data that's already local (journal + closes)."""
    try:
        from . import forwardtest
        jdf = journal.read(jpath, limit=10000)
        closes = pd.DataFrame({k: v["close"] for k, v in bars.items()})
        rep = forwardtest.report(forwardtest.signal_events(jdf), closes, horizon=horizon,
                                 session_cfg=cfg)
        return rep.get("ic"), rep.get("n", 0)
    except Exception:
        return None, 0


# ---------------------------------------------------------------- data health
def data_health(bars: dict, macro, panel, genome: Genome, asset_map: dict, day,
                max_nan=0.3) -> dict:
    """Pre-trade sanity check. Every fetch in datafeed fails *silently* (returns empty), so
    without this a rate-limited run still trains, predicts and trades on median-filled
    placeholders. A stock with no bar for this session (weekend/holiday) is simply not tradable,
    not an issue — unless other stocks do have one, which means its own feed failed."""
    issues, tradable = [], []
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    stocks_with_bar = [x for x, df in bars.items()
                       if asset_map.get(x) == "stock" and df.index[-1] >= day]
    macro_ok = True
    if genome.use_macro:
        mm = macro.dropna(how="all") if macro is not None else pd.DataFrame()
        if (mm.empty or (day - mm.index[-1]).days > 5
                or mm.iloc[-1].reindex(["vix_lvl", "spy_ret21"]).isna().any()):
            macro_ok = False
            issues.append("macro data missing or stale — holding every position")
    for sym, df in bars.items():
        last = df.index[-1]
        if asset_map.get(sym) == "stock" and last < day:
            if stocks_with_bar:
                issues.append(f"{sym}: no bar for {day:%Y-%m-%d} while other stocks have one")
            continue
        if asset_map.get(sym) != "stock" and (day - last).days > 1:
            issues.append(f"{sym}: last bar {last:%Y-%m-%d} is stale")
            continue
        row = panel[panel["symbol"] == sym].tail(1)
        nan_share = float(row[cols].isna().mean(axis=1).iloc[0]) if cols and not row.empty else 0.0
        if nan_share > max_nan:
            issues.append(f"{sym}: {nan_share:.0%} of features missing")
            continue
        if macro_ok:
            tradable.append(sym)
    return {"tradable": tradable, "issues": issues,
            "sources": {x: df.attrs.get("source") for x, df in bars.items()}}


# ---------------------------------------------------------------- strategy lab
def _search_and_maybe_promote(s, cfg, panel, genome, jpath, trigger, live_ic=None, n_obs=0,
                              exclude_groups=(), exec_open=()):
    """Run the Strategy Lab (search + champion/challenger comparison) and promote if the
    challenger clears promote_margin AND promote_min_score. Champion and challengers are scored
    by the same function with the same costs, execution model and sizing. Never trades."""
    evo = cfg.get("evolution", {})
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=365 * evo.get("eval_years", 2))
    pe = panel[panel["date"] >= cutoff]
    cost = cfg.get("fee_bps", 2) + cfg.get("slippage_bps", 5)
    sizing = cfg.get("sizing", "equal")
    # Vary the seed per calendar day so each run explores new territory; still reproducible.
    seed = evo.get("seed") or int(pd.Timestamp.now("UTC").strftime("%Y%m%d"))

    if evo.get("method", "evolution") == "optuna":
        from . import optimize
        storage = f"sqlite:///{os.path.join(s.data_dir, 'optuna.db')}"   # warm-starts weekly
        rows, _, _ = optimize.optimize(pe, n_trials=evo.get("trials", 25), seed=seed,
                                       cost_bps=cost, storage=storage, exclude_groups=exclude_groups,
                                       exec_open=exec_open, sizing=sizing)
    else:
        rows, _ = models.evolve(pe, pop_size=evo.get("pop_size", 6),
                                generations=evo.get("generations", 2), cost_bps=cost, seed=seed,
                                exclude_groups=exclude_groups, exec_open=exec_open, sizing=sizing)
    try:
        champ_score, _ = models.score_genome(pe, genome, cost, exec_open, sizing)
    except Exception:
        champ_score = -999.0
    challenger = rows[0]
    min_score = evo.get("promote_min_score", 0.0)
    min_exposure = evo.get("promote_min_exposure", 0.10)
    event = {"trigger": trigger, "seed": seed, "champion": genome.name,
             "champion_score": round(champ_score, 3),
             "champion_live_ic": (round(live_ic, 3) if live_ic is not None else None),
             "champion_live_obs": n_obs,
             "challenger": challenger["name"], "challenger_score": challenger["score"],
             "challenger_exposure": challenger.get("exposure"),
             "excluded": list(exclude_groups), "sizing": sizing}
    ch_exposure = challenger.get("exposure") or 0.0
    if (challenger["score"] > champ_score + evo.get("promote_margin", 0.05)
            and challenger["score"] > min_score and ch_exposure >= min_exposure):
        g = challenger["genome"]
        save_champion(s, g, models.train_final_model(panel, g))
        event["promoted"] = g.name
    else:
        event["promoted"] = None
        event["promote_blocked_reason"] = (
            "below promote_min_score" if challenger["score"] <= min_score else
            "below promote_min_exposure" if ch_exposure < min_exposure else None)
    journal.log(jpath, "evolution", event)
    return event


def _sentiment(cfg, s, symbols, update: bool):
    """(sentiment features, days of history, ready?). Sentiment only counts once the accumulated
    store spans the evaluation window; before that it is ~zero in training/backtests but real
    live — a train/serve skew the lab can't see — so it's switched off everywhere."""
    evo = cfg.get("evolution", {})
    path = os.path.join(s.data_dir, "sentiment.csv")
    store = (datafeed.update_sentiment_store(symbols, s, path, cfg.get("sentiment_backend", "auto"))
             if update else datafeed.load_sentiment_store(path))
    days = datafeed.sentiment_coverage_days(store)
    need = evo.get("sentiment_min_days", 365 * evo.get("eval_years", 2))
    return datafeed.sentiment_features(store, symbols), days, days >= need


def _effective(genome: Genome, sent_ready: bool) -> Genome:
    return genome if sent_ready or not genome.use_sentiment else replace(genome, use_sentiment=False)


def _exclusions(cfg, sent_ready: bool) -> tuple:
    ex = tuple(cfg.get("evolution", {}).get("exclude_feature_groups", []))
    return ex if sent_ready else ex + ("sentiment",)


def check_intraday(config_path="universe.json") -> dict | None:
    """Lightweight IC health check (manual trigger). Step 1 is cheap (closes + the journal, no
    retraining); it only escalates to a Strategy Lab run if the live IC has genuinely degraded
    and the lab hasn't already run this session. The daily run() performs the same check."""
    s = load_settings()
    cfg = json.load(open(config_path))
    evo = cfg.get("evolution", {})
    if not (evo.get("enabled") and evo.get("intraday_reaction", True)):
        return None
    jpath = os.path.join(s.data_dir, "journal.db")
    day = session.as_of(cfg)
    if evolved_on(jpath, cfg, day):
        return None

    stocks, _, symbols, asset_map = _universe(cfg)
    genome = load_champion(s)
    short_start = dt.datetime.utcnow() - dt.timedelta(days=400)
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, short_start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 20}
    if not bars:
        return None
    live_ic, n_obs = compute_live_ic(jpath, bars, genome.horizon, cfg)
    panic = (live_ic is not None and n_obs >= evo.get("ic_min_obs", 60)
             and live_ic < evo.get("ic_panic_threshold", -0.02))
    if not panic:
        return None

    start = dt.datetime.utcnow() - dt.timedelta(days=365 * cfg.get("years", 4))
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 120}
    macro = datafeed.get_macro(start, s)
    sent, _, sent_ready = _sentiment(cfg, s, symbols, update=False)
    exo = exodata.get_exo(start, os.path.join(s.data_dir, "snapshots.csv"))
    full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
    panel = build_panel(bars, macro, sent, full, exo)
    return _search_and_maybe_promote(s, cfg, panel, _effective(genome, sent_ready), jpath,
                                     "ic_panic", live_ic, n_obs, _exclusions(cfg, sent_ready), stocks)


def run(config_path="universe.json", force_evolve=False, no_trade=False,
        promote: Genome | None = None, skip_if_done=False) -> dict:
    s = load_settings()
    cfg = json.load(open(config_path))
    evo = cfg.get("evolution", {})
    stocks, cryptos, symbols, asset_map = _universe(cfg)
    start = dt.datetime.utcnow() - dt.timedelta(days=365 * cfg.get("years", 4))
    jpath = os.path.join(s.data_dir, "journal.db")
    day = session.as_of(cfg)
    out = {"ts": pd.Timestamp.now("UTC").isoformat(), "session": f"{day:%Y-%m-%d}", "symbols": symbols}
    if skip_if_done and ran_for_session(jpath, cfg, day):
        return {**out, "skipped": "this session already has an after-close run"}

    # 1. data
    bars = {k: v for k, v in datafeed.get_daily_bars(symbols, asset_map, start,
                                                     dt.datetime.utcnow(), s).items() if len(v) > 120}
    macro = datafeed.get_macro(start, s)
    sent, sent_days, sent_ready = _sentiment(cfg, s, symbols, update=True)
    snaps = exodata.collect_snapshots(cfg.get("snapshot_symbols", symbols), asset_map,
                                      os.path.join(s.data_dir, "snapshots.csv"))
    exo = {"options_market": exodata.options_market_history(start),
           "onchain": exodata.onchain_history(), "snapshots": snaps}

    # 2. champion (+ a promotion requested from the UI) and latest signals
    full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
    panel = build_panel(bars, macro, sent, full, exo)
    genome = load_champion(s)
    if promote is not None:
        save_champion(s, promote, models.train_final_model(panel, _effective(promote, sent_ready)))
        journal.log(jpath, "evolution", {"trigger": "manual_promote", "champion": genome.name,
                                         "promoted": promote.name})
        genome = promote
    live = _effective(genome, sent_ready)
    model = models.train_final_model(panel, live)
    recent = models.recent_signals(panel, live, model)
    latest = recent.iloc[-1]
    health = data_health(bars, macro, panel, live, asset_map, day)
    out["genome"] = genome.name
    out["signals"] = _clean({k: round(float(v), 4) for k, v in latest.items()})
    out["tradable"] = health["tradable"]
    out["data_health"] = {"issues": health["issues"], "sources": health["sources"]}
    out["sentiment_days"] = sent_days
    if live.use_sentiment != genome.use_sentiment:
        out["sentiment_note"] = (f"sentiment off: {sent_days} days of history, needs "
                                 f"{evo.get('sentiment_min_days', 365 * evo.get('eval_years', 2))}")
    if not macro.empty:
        m = macro.iloc[-1]
        out["regime"] = _clean({"vix": round(float(m.get("vix_lvl", np.nan)), 2),
                                "vix_z": round(float(m.get("vix_z", np.nan)), 2),
                                "spy_ret21": round(float(m.get("spy_ret21", np.nan)), 4),
                                "usdidr": round(float(m.get("usdidr_lvl", np.nan)), 1),
                                "usdidr_chg5": round(float(m.get("usdidr_chg5", np.nan)), 4)})
    json.dump(out, open(os.path.join(s.data_dir, "latest_signals.json"), "w"),
              indent=2, default=str)
    # the forward test only gets genuinely new observations: a stock's carried-over value on a
    # weekend, or a symbol with broken data, is logged as None instead of a duplicate/garbage row
    journal.log(jpath, "signals", {k: (v if k in health["tradable"] else None)
                                   for k, v in out["signals"].items()})
    if health["issues"]:
        journal.log(jpath, "data_health", health)

    # 3. paper-trade rebalance of the tradable symbols only (Alpaca queues stock orders placed
    #    after close for the next open; crypto fills now). Untradable symbols keep their position.
    if cfg.get("trade", True) and not no_trade and datafeed.has_alpaca(s):
        tc = datafeed.trading_client(s)
        try:
            out["market_open"] = bool(tc.get_clock().is_open)
        except Exception:
            out["market_open"] = None
        closes = pd.DataFrame({k: v["close"] for k, v in bars.items()}).sort_index().ffill()
        w = bt.target_weights(recent, closes, live.max_leverage, cfg.get("sizing", "equal"),
                              allow_short=cfg.get("allow_short", False)).iloc[-1]
        for sym in cryptos:                                   # Alpaca: no crypto shorts
            if sym in w:
                w[sym] = max(w[sym], 0.0)
        equity = float(tc.get_account().equity)
        targets = {sym: float(w[sym]) * equity for sym in health["tradable"] if sym in w}
        if targets:
            results = datafeed.rebalance(tc, targets, asset_map)
            journal.log(jpath, "orders", results)
            out["orders"] = results

    # 4. autonomous Strategy Lab: scheduled (session weekday / catch-up) + live-IC-triggered
    if promote is None:
        live_ic, n_obs = compute_live_ic(jpath, bars, genome.horizon, cfg)
        due = lab_due(jpath, cfg, day)
        ic_panic = (bool(evo.get("enabled")) and not evolved_on(jpath, cfg, day)
                    and live_ic is not None and n_obs >= evo.get("ic_min_obs", 60)
                    and live_ic < evo.get("ic_panic_threshold", -0.02))
        if force_evolve or due or ic_panic:
            trigger = "manual" if force_evolve else ("ic_panic" if ic_panic and not due else "scheduled")
            out["evolution"] = _search_and_maybe_promote(
                s, cfg, panel, live, jpath, trigger, live_ic, n_obs,
                _exclusions(cfg, sent_ready), stocks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe.json")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--no-trade", action="store_true")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit early if this session already has an after-close run "
                         "(for scheduled/backup crons; GitHub cron is late and sometimes doubled)")
    ap.add_argument("--promote-genome", default="",
                    help="genome JSON to install as champion before the run (from the UI)")
    ap.add_argument("--backfill-sentiment", type=int, default=0, metavar="DAYS",
                    help="only build the sentiment store DAYS back, then exit")
    ap.add_argument("--check-intraday", action="store_true",
                    help="run only the cheap IC health check (+ Strategy Lab if it panics)")
    a = ap.parse_args()
    if a.backfill_sentiment:
        s, cfg = load_settings(), json.load(open(a.config))
        store = datafeed.backfill_sentiment_store(
            _universe(cfg)[2], s, os.path.join(s.data_dir, "sentiment.csv"),
            cfg.get("sentiment_backend", "auto"), days=a.backfill_sentiment)
        print(json.dumps({"status": "ok", "sentiment_days": datafeed.sentiment_coverage_days(store)}))
    elif a.check_intraday:
        event = check_intraday(a.config)
        print(json.dumps({"status": "ok", "evolution": event}, default=str))
    else:
        promote = None
        if a.promote_genome.strip():
            d = json.loads(a.promote_genome)
            d["feature_groups"] = tuple(d.get("feature_groups") or ("momentum",))
            promote = Genome(**d)
        print(json.dumps({"status": "ok", **run(a.config, a.evolve, a.no_trade, promote,
                                                a.skip_if_done)}, default=str))


if __name__ == "__main__":
    main()
