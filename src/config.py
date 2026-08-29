from __future__ import annotations

import os
from dataclasses import dataclass


def get_secret(key: str, default: str = "") -> str:
    """Env var first (Docker/.env), then st.secrets (Streamlit Cloud)."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    alpaca_key: str
    alpaca_secret: str
    paper: bool
    fred_key: str
    sentiment_backend: str = "auto"
    data_dir: str = "data"
    model_dir: str = "models"


def load_settings() -> Settings:
    return Settings(
        alpaca_key=get_secret("APCA_API_KEY_ID"),
        alpaca_secret=get_secret("APCA_API_SECRET_KEY"),
        paper=get_secret("APCA_PAPER", "true").strip().lower() != "false",
        fred_key=get_secret("FRED_API_KEY"),
        sentiment_backend=get_secret("SENTIMENT_BACKEND", "auto"),
        data_dir=get_secret("DATA_DIR", "data"),
        model_dir=get_secret("MODEL_DIR", "models"),
    )
