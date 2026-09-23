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


def fitness(m: dict) -> float:
    """The single score every strategy is ranked by — challengers in the search AND the sitting
    champion, so promotion compares like with like."""
    score = m["sharpe"] - 0.5 * abs(m["max_drawdown"])
    score += 0.5 * min(m["exposure"], 0.4)       # reward being active, capped so it can't dominate
    return float(score - (0.5 if m["exposure"] < 0.05 else 0.0))


def target_weights(signals: pd.DataFrame, prices: pd.DataFrame, max_leverage=1.0,
                   sizing="equal", allow_short=True, hold_until=None, vol_window=60) -> pd.DataFrame:
    """Portfolio weights decided at each date's close (callers shift for execution).
    equal:   weight = signal * max_leverage / n_assets
    inv_vol: same budget tilted toward calmer assets — scale_i = (1/vol_i) / mean(1/vol),
             clipped to [0.25, 4], then gross exposure capped at max_leverage.
    hold_until: optional bool frame (True = asset trades that day); on other days the previous
             decision is kept, e.g. stocks over weekends while crypto keeps moving."""
    prices = prices.sort_index().ffill()
    sig = signals.reindex(prices.index).ffill().fillna(0.0).clip(-1, 1)
    if not allow_short:
        sig = sig.clip(lower=0)
    w = sig * (max_leverage / max(1, sig.shape[1]))
    if sizing == "inv_vol":
        inv = 1 / prices.pct_change().rolling(vol_window, min_periods=20).std().replace(0, np.nan)
        w = w * inv.div(inv.mean(axis=1), axis=0).clip(0.25, 4).fillna(1.0)
        w = w.div((w.abs().sum(axis=1) / max_leverage).clip(lower=1.0), axis=0)
    if hold_until is not None:
        cols = [c for c in hold_until.columns if c in w]
        w[cols] = w[cols].where(hold_until[cols].reindex(w.index).fillna(False).astype(bool)).ffill().fillna(0.0)
    return w


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, fee_bps=5.0,
                 slippage_bps=5.0, max_leverage=1.0, allow_short=True,
                 opens: pd.DataFrame | None = None, exec_open=(), sizing="equal") -> BacktestResult:
    """Decide at close t, trade on the next bar (no lookahead).
    exec_open: symbols whose orders fill at the NEXT OPEN, as the live pipeline's after-close
    orders do for stocks — they earn open-to-open returns on their own trading calendar instead
    of close-to-close, so the overnight gap the live account can't capture isn't counted.
    Everything else (crypto) earns close-to-close."""
    prices = prices.sort_index().ffill()
    rets = prices.pct_change().fillna(0.0)
    cols = [c for c in exec_open if opens is not None and c in opens and c in rets]
    trades = None
    if cols:
        o = opens.reindex(prices.index)[cols]
        trades = o.notna()
        rets[cols] = o.apply(lambda s: (s.dropna().shift(-1) / s.dropna() - 1)
                             .reindex(o.index)).fillna(0.0)
    w = target_weights(signals, prices, max_leverage, sizing, allow_short, trades)
    w = w.shift(1).fillna(0.0)
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
