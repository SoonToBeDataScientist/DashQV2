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


# Tunable from universe.json -> "fitness": {...} (see configure_fitness). Defaults:
#   return_weight   reward per unit of CAGR (capped at 100%): being invested only pays when it makes money
#   dd_weight       penalty per unit of max drawdown
#   dd_budget       the drawdown you are happy to live with (fraction, 0.12 = -12%)
#   dd_over_weight  EXTRA penalty per unit of drawdown beyond the budget (the brake on over-sizing)
#   min_exposure    below this average exposure a strategy is considered "doing nothing" (-0.5)
FITNESS_DEFAULTS = {"return_weight": 2.0, "dd_weight": 0.5, "dd_budget": 0.12,
                    "dd_over_weight": 4.0, "min_exposure": 0.05}
FITNESS = dict(FITNESS_DEFAULTS)


def configure_fitness(cfg: dict | None = None) -> dict:
    """Reset to the defaults, then apply overrides from the config's "fitness" block. Unknown keys
    and non-numeric values are ignored so a typo can't break the daily pipeline."""
    FITNESS.clear(); FITNESS.update(FITNESS_DEFAULTS)
    for k, v in (cfg or {}).items():
        if k in FITNESS_DEFAULTS:
            try:
                FITNESS[k] = float(v)
            except (TypeError, ValueError):
                pass
    return dict(FITNESS)


def fitness(m: dict) -> float:
    """The single score every strategy is ranked by — challengers in the search AND the sitting
    champion, so promotion compares like with like.

    Sharpe alone can't tell a tiny bet from a big one (it doesn't change when you size a position
    up), so the old score all but ignored exposure. This one rewards what the strategy actually
    earns (CAGR) and charges for drawdown, with a steep extra charge once drawdown passes
    `dd_budget`. Scaling a good strategy up therefore pays until its drawdown reaches the budget
    and then stops paying: more exposure, but only as far as the risk you accept, and only if the
    returns justify it."""
    f = FITNESS
    dd = abs(m["max_drawdown"])
    score = (m["sharpe"] + f["return_weight"] * min(m["cagr"], 1.0)
             - f["dd_weight"] * dd - f["dd_over_weight"] * max(0.0, dd - f["dd_budget"]))
    return float(score - (0.5 if m["exposure"] < f["min_exposure"] else 0.0))


def parse_sizing(sizing: str):
    """'equal' | 'inv_vol', optionally followed by '+vt<N>' (e.g. 'equal+vt10') = also scale the
    book to an N% annualized portfolio-volatility target. -> (base, target_vol or None).
    A malformed suffix falls back to no target rather than breaking the daily pipeline."""
    base, _, vt = str(sizing).partition("+vt")
    try:
        tv = float(vt) / 100 if vt else None
    except ValueError:
        tv = None
    return (base or "equal"), (tv if tv and tv > 0 else None)


def _vol_target(w: pd.DataFrame, prices: pd.DataFrame, target: float, max_leverage: float,
                window=60, max_scale=3.0, ann=252) -> pd.DataFrame:
    """Scale each date's weights so the portfolio they form has ~`target` annualized volatility,
    using only returns up to that date (the weights are shifted for execution by the caller).
    The risk measured is the *position's* (w' Σ w, Σ = trailing `window`-day covariance of asset
    returns), not the strategy's past P&L, so a strategy that has been flat for weeks is still
    sized correctly when it wakes up. The scale is capped at `max_scale` (a nearly flat book
    isn't blown up without limit) and gross exposure never exceeds `max_leverage`. Same
    convention as the backtest's "ann vol" (252 on the calendar-day series)."""
    cols = [c for c in w.columns if c in prices.columns]
    if not cols or w.empty:
        return w
    R = prices.pct_change().fillna(0.0)[cols].reindex(w.index).fillna(0.0).to_numpy(float)
    W = w[cols].to_numpy(float)
    scale = np.ones(len(w))
    for i in range(19, len(w)):                          # need ~a month of returns for a covariance
        sub = R[max(0, i - window + 1): i + 1]
        var = float(W[i] @ np.atleast_2d(np.cov(sub, rowvar=False)) @ W[i]) * ann
        if var > 1e-12:
            scale[i] = min(max_scale, target / np.sqrt(var))
        elif np.abs(W[i]).sum() > 0:
            scale[i] = max_scale
    out = w.mul(scale, axis=0)
    gross = out.abs().sum(axis=1)
    return out.div((gross / max_leverage).clip(lower=1.0), axis=0)


def target_weights(signals: pd.DataFrame, prices: pd.DataFrame, max_leverage=1.0,
                   sizing="equal", allow_short=True, hold_until=None, vol_window=60) -> pd.DataFrame:
    """Portfolio weights decided at each date's close (callers shift for execution).
    equal:   weight = signal * max_leverage / n_assets
    inv_vol: same budget tilted toward calmer assets — scale_i = (1/vol_i) / mean(1/vol),
             clipped to [0.25, 4], then gross exposure capped at max_leverage.
    +vt<N>:  (suffix, either base) after the above, scale the whole book to an N% annualized
             volatility target — up in calm markets, down in turbulent ones — see _vol_target.
    hold_until: optional bool frame (True = asset trades that day); on other days the previous
             decision is kept, e.g. stocks over weekends while crypto keeps moving."""
    sizing, target_vol = parse_sizing(sizing)
    prices = prices.sort_index().ffill()
    sig = signals.reindex(prices.index).ffill().fillna(0.0).clip(-1, 1)
    if not allow_short:
        sig = sig.clip(lower=0)
    w = sig * (max_leverage / max(1, sig.shape[1]))
    if sizing == "inv_vol":
        inv = 1 / prices.pct_change().rolling(vol_window, min_periods=20).std().replace(0, np.nan)
        w = w * inv.div(inv.mean(axis=1), axis=0).clip(0.25, 4).fillna(1.0)
        w = w.div((w.abs().sum(axis=1) / max_leverage).clip(lower=1.0), axis=0)
    if target_vol:
        w = _vol_target(w, prices, target_vol, max_leverage, vol_window)
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
