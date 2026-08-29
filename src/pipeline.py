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
    os.makedirs(s.model_dir, exist_ok=True)
    json.dump(asdict(genome), open(os.path.join(s.model_dir, "champion_genome.json"), "w"), indent=2)
    if model is not None:
        models.save_bundle(os.path.join(s.model_dir, "champion.joblib"), genome, model)


def run(config_path="universe.json", force_evolve=False, no_trade=False) -> dict:
    s = load_settings()
    cfg = json.load(open(config_path))
    stocks, cryptos = cfg.get("stocks", []), cfg.get("cryptos", [])
    symbols = stocks + cryptos
    asset_map = {**{x: "stock" for x in stocks}, **{x: "crypto" for x in cryptos}}
    start = dt.datetime.utcnow() - dt.timedelta(days=365 * cfg.get("years", 4))
    jpath = os.path.join(s.data_dir, "journal.db")
    out = {"ts": dt.datetime.utcnow().isoformat(), "symbols": symbols}

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

    # 4. weekly champion/challenger evolution
    evo = cfg.get("evolution", {})
    if force_evolve or (evo.get("enabled") and dt.datetime.utcnow().weekday() == int(evo.get("weekday", 5))):
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * evo.get("eval_years", 2))
        pe = panel[panel["date"] >= cutoff]
        rows, _ = models.evolve(pe, pop_size=evo.get("pop_size", 6),
                                generations=evo.get("generations", 2),
                                cost_bps=cfg.get("fee_bps", 2) + cfg.get("slippage_bps", 5))
        prices = pe.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
        champ_sig, _ = models.walk_forward_signals(pe, genome)
        cm = bt.run_backtest(prices, champ_sig, cfg.get("fee_bps", 2) / 2,
                             cfg.get("slippage_bps", 5) / 2, genome.max_leverage).metrics
        champ_score = cm["sharpe"] - 0.5 * abs(cm["max_drawdown"])
        challenger = rows[0]
        event = {"champion": genome.name, "champion_score": round(champ_score, 3),
                 "challenger": challenger["name"], "challenger_score": challenger["score"]}
        if challenger["score"] > champ_score + evo.get("promote_margin", 0.05):
            g = challenger["genome"]
            save_champion(s, g, models.train_final_model(panel, g))
            event["promoted"] = g.name
        else:
            event["promoted"] = None
        journal.log(jpath, "evolution", event)
        out["evolution"] = event
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe.json")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--no-trade", action="store_true")
    a = ap.parse_args()
    print(json.dumps({"status": "ok", **run(a.config, a.evolve, a.no_trade)}, default=str))


if __name__ == "__main__":
    main()
