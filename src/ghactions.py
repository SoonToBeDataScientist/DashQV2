"""Trigger GitHub Actions from the Streamlit UI.

On Streamlit Cloud the app's filesystem is not the repo: a champion saved there, or an order
journaled there, never reaches the pipeline and disappears on restart. With GITHUB_TOKEN (a
fine-grained token with Actions: write on this repo) and GITHUB_REPO ("owner/name") set, the UI
asks the pipeline to do the work instead, so git stays the single source of truth.
"""
from __future__ import annotations

import requests

from .config import get_secret


def configured() -> bool:
    return bool(get_secret("GITHUB_TOKEN") and get_secret("GITHUB_REPO"))


def actions_url(workflow: str) -> str:
    return f"https://github.com/{get_secret('GITHUB_REPO')}/actions/workflows/{workflow}"


def dispatch(workflow: str, inputs: dict | None = None) -> None:
    fmt = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
    r = requests.post(
        f"https://api.github.com/repos/{get_secret('GITHUB_REPO')}/actions/workflows/{workflow}/dispatches",
        headers={"Authorization": f"Bearer {get_secret('GITHUB_TOKEN')}",
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        json={"ref": get_secret("GITHUB_REF", "main"),
              "inputs": {k: fmt(v) for k, v in (inputs or {}).items()}},
        timeout=15)
    if r.status_code != 204:
        raise RuntimeError(f"GitHub dispatch failed ({r.status_code}): {r.text[:300]}")
