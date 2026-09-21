"""Which trading session a timestamp belongs to.

GitHub cron fires hours late (a 21:30 UTC schedule routinely lands 23:00-00:00 UTC), so the
wall-clock UTC date is the wrong key for "which day was this run for". A run belongs to the
session that is open or most recently closed: everything from one open to the next open maps
to the same date, in the exchange's own timezone (so DST is handled).
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT = {"tz": "America/New_York", "open": "09:30", "close": "16:00"}


def _cfg(cfg: dict | None):
    ss = {**DEFAULT, **((cfg or {}).get("session") or {})}
    hm = lambda s: pd.Timedelta(hours=int(s.split(":")[0]), minutes=int(s.split(":")[1]))
    return ZoneInfo(ss["tz"]), hm(ss["open"]), hm(ss["close"])


def _utc(ts) -> pd.Series:
    """Naive-UTC or tz-aware timestamps (scalar or Series) -> tz-aware UTC Series."""
    s = pd.Series(pd.to_datetime(ts if isinstance(ts, (pd.Series, pd.Index, list)) else [ts]))
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def session_dates(cfg: dict | None, ts) -> pd.Series:
    """Session date (naive, midnight) for each timestamp."""
    tz, open_, _ = _cfg(cfg)
    local = _utc(ts).dt.tz_convert(tz) - open_
    return pd.to_datetime(local.dt.date)


def as_of(cfg: dict | None, ts=None) -> pd.Timestamp:
    """Session date for one timestamp (default: now)."""
    return session_dates(cfg, pd.Timestamp.now("UTC") if ts is None else ts).iloc[0]


def close_utc(cfg: dict | None, day) -> pd.Timestamp:
    """That session's close as naive UTC."""
    tz, _, close = _cfg(cfg)
    return (pd.Timestamp(day).normalize().tz_localize(tz) + close).tz_convert("UTC").tz_localize(None)
