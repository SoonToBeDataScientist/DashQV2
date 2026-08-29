from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd


def _conn(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS events (ts TEXT, kind TEXT, payload TEXT)")
    return c


def log(path: str, kind: str, payload):
    with _conn(path) as c:
        c.execute("INSERT INTO events VALUES (?,?,?)",
                  (pd.Timestamp.utcnow().isoformat(), kind, json.dumps(payload, default=str)))


def read(path: str, kind: str | None = None, limit=500) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts", "kind", "payload"])
    with _conn(path) as c:
        if kind:
            df = pd.read_sql("SELECT * FROM events WHERE kind=? ORDER BY ts DESC LIMIT ?",
                             c, params=(kind, limit))
        else:
            df = pd.read_sql("SELECT * FROM events ORDER BY ts DESC LIMIT ?", c, params=(limit,))
    return df
