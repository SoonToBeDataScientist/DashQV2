from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    drawdown: pd.Series
    metrics: dict


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, fee_bps=5.0,
                 slippage_bps=5.0, max_leverage=1.0, allow_short=True) -> BacktestResult:
    """Signal value IS the position fraction: weight = signal * (max_leverage / n_assets)."""
    prices = prices.sort_index().ffill()
    rets = prices.pct_change().fillna(0.0)
    sig = signals.reindex(prices.index).fillna(0.0).clip(-1, 1)
    if not allow_short:
        sig = sig.clip(lower=0)
    sig = sig.shift(1).fillna(0.0)                      # trade on next bar (no lookahead)
    w = sig * (max_leverage / max(1, sig.shape[1]))
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    net = (w * rets).sum(axis=1) - turnover * (fee_bps + slippage_bps) / 1e4
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    return BacktestResult(equity, net, w, dd, _metrics(net, equity, dd, w, turnover))


def _metrics(r, equity, dd, w, turnover) -> dict:
    n, ann = max(len(r), 1), 252
    cagr = equity.iloc[-1] ** (ann / n) - 1 if equity.iloc[-1] > 0 else -1.0
    maxdd = float(dd.min())
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(cagr),
        "ann_vol": float(r.std() * np.sqrt(ann)),
        "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(ann)),
        "sortino": float(r.mean() / (r[r < 0].std() + 1e-12) * np.sqrt(ann)),
        "max_drawdown": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "exposure": float(w.abs().sum(axis=1).mean()),
        "ann_turnover": float(turnover.mean() * ann),
    }
