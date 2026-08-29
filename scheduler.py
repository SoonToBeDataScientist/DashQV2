from __future__ import annotations

import datetime as dt
import json
import os
import traceback

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src import datafeed, exodata, journal, models
from src.config import load_settings
from src.features import build_live_row, feature_columns
from src.pipeline import run as run_pipeline

CONFIG = os.environ.get("UNIVERSE_CONFIG", "universe.json")


def _journal_path():
    return os.path.join(load_settings().data_dir, "journal.db")


def safe(fn, name):
    try:
        fn()
    except Exception:
        journal.log(_journal_path(), "error", {"job": name, "trace": traceback.format_exc()})


def daily_job():
    safe(lambda: run_pipeline(CONFIG), "daily")


def realtime_snapshot_job():
    """Log realtime signals+prices -> raw material for the continuous forward test."""
    def _run():
        s = load_settings()
        cfg = json.load(open(CONFIG))
        stocks, cryptos = cfg.get("stocks", []), cfg.get("cryptos", [])
        symbols = stocks + cryptos
        asset_map = {**{x: "stock" for x in stocks}, **{x: "crypto" for x in cryptos}}
        bundle = models.load_bundle(os.path.join(s.model_dir, "champion.joblib"))
        if not bundle:
            return
        genome, model = bundle["genome"], bundle["model"]
        clock = None
        if datafeed.has_alpaca(s):
            try:
                clock = datafeed.trading_client(s).get_clock()
            except Exception:
                pass
        start = dt.datetime.utcnow() - dt.timedelta(days=400)
        bars = datafeed.get_daily_bars(symbols, asset_map, start, dt.datetime.utcnow(), s)
        macro = datafeed.get_macro(start, s)
        sent = datafeed.get_news_sentiment(symbols, s,
                                           backend=cfg.get("sentiment_backend", "auto"), days=10)
        exo = exodata.get_exo(start, os.path.join(s.data_dir, "snapshots.csv"))
        payload = {}
        for sym, df in bars.items():
            if asset_map.get(sym) == "stock" and clock is not None and not clock.is_open:
                continue                                   # skip stocks outside market hours
            intr = datafeed.get_intraday_bars(sym, asset_map[sym], s)
            row = build_live_row(sym, df, intr, macro, sent, genome, exo)
            cols = [c for c in feature_columns(genome) if c in row.columns]
            payload[sym] = {"signal": round(float(model.predict_signal(row[cols])[0]), 4),
                            "price": float(intr["close"].iloc[-1]) if not intr.empty else None}
        if payload:
            journal.log(_journal_path(), "realtime", payload)
    safe(_run, "realtime")


def catch_up_on_start():
    """If the container was down through the scheduled run, do it now."""
    try:
        df = journal.read(_journal_path(), kind="signals", limit=1)
        stale = df.empty or (pd.Timestamp.utcnow().tz_localize(None)
                             - pd.Timestamp(df["ts"].iloc[0])) > pd.Timedelta(hours=20)
        if stale:
            daily_job()
    except Exception:
        pass


if __name__ == "__main__":
    sch = BlockingScheduler(timezone="UTC")
    # EOD pipeline daily 21:30 UTC (after US close; crypto weekends included)
    sch.add_job(daily_job, CronTrigger(hour=21, minute=30), id="daily",
                misfire_grace_time=7200, coalesce=True)
    # realtime forward-test snapshots every 30 min during US market hours
    sch.add_job(realtime_snapshot_job, CronTrigger(day_of_week="mon-fri", hour="13-20",
                minute="*/30"), id="rt", misfire_grace_time=900, coalesce=True)
    catch_up_on_start()
    print("Scheduler started:", sch.get_jobs(), flush=True)
    sch.start()
