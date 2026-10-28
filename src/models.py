from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .backtest import fitness, run_backtest, window_metrics
from .features import FEATURE_GROUPS, feature_columns

try:
    import lightgbm  # noqa: F401
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


# ---------------------------------------------------------------- genome
@dataclass
class Genome:
    name: str = "base"
    model_type: str = "lgbm"                 # lgbm | hgb | rf
    feature_groups: tuple = ("trend", "momentum", "volatility")
    use_macro: bool = True
    use_sentiment: bool = True
    horizon: int = 5                         # forward-return days
    train_window: int = 504                  # rolling training window (days)
    retrain_every: int = 21
    smooth: int = 3                          # EMA span on raw signal
    entry: float = 0.15                      # dead zone |s| < entry -> 0 (linear remap)
    max_leverage: float = 1.0
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4

    def key(self) -> str:
        d = asdict(self)
        d.pop("name", None)
        return json.dumps(d, sort_keys=True)


# "macro" and "sentiment" aren't FEATURE_GROUPS but can be excluded the same way — e.g.
# sentiment while its accumulated history is still too short to train on.
def _allowed(genome_dict: dict, exclude_groups) -> dict:
    for k in ("macro", "sentiment"):
        if k in exclude_groups:
            genome_dict[f"use_{k}"] = False
    return genome_dict


def random_genome(rng, name="", exclude_groups=()) -> Genome:
    eligible = [g for g in FEATURE_GROUPS if g not in exclude_groups]
    groups = [g for g in eligible if rng.random() < 0.6] or ["momentum"]
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    return Genome(**_allowed(dict(
        name=name, model_type=pick(["lgbm", "lgbm", "hgb", "rf"]),
        feature_groups=tuple(groups),
        use_macro=bool(rng.random() < 0.7),
        use_sentiment=bool(rng.random() < 0.6),
        horizon=int(pick([3, 5, 10])), train_window=int(pick([252, 378, 504])),
        retrain_every=int(pick([10, 21, 42])), smooth=int(pick([1, 2, 3, 5, 8])),
        entry=float(np.round(rng.uniform(0.02, 0.15), 3)),
        max_leverage=float(pick([0.5, 1.0, 1.5])),
        n_estimators=int(pick([150, 300])),
        learning_rate=float(pick([0.02, 0.05, 0.1])),
        max_depth=int(pick([3, 4, 5, 6]))), exclude_groups))


def crossover(a: Genome, b: Genome, rng) -> Genome:
    da, db = asdict(a), asdict(b)
    d = {k: (da[k] if rng.random() < 0.5 else db[k]) for k in da if k != "name"}
    d["feature_groups"] = tuple(d["feature_groups"]) or ("momentum",)
    return Genome(name="", **d)


def mutate(g: Genome, rng, exclude_groups=()) -> Genome:
    d = asdict(g)
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    if rng.random() < 0.5:
        groups = list(d["feature_groups"])
        if rng.random() < 0.5 and len(groups) > 1:
            groups.pop(int(rng.integers(len(groups))))
        else:
            cand = [x for x in FEATURE_GROUPS if x not in groups and x not in exclude_groups]
            if cand:
                groups.append(pick(cand))
        d["feature_groups"] = tuple(groups) or ("momentum",)
    if rng.random() < 0.4:
        d["entry"] = float(np.clip(d["entry"] * rng.lognormal(0, 0.3), 0.02, 0.4))
    for k, opts in [("horizon", [3, 5, 10]), ("train_window", [252, 378, 504]),
                    ("retrain_every", [10, 21, 42]), ("smooth", [1, 2, 3, 5, 8]),
                    ("max_leverage", [0.5, 1.0, 1.5]), ("max_depth", [3, 4, 5, 6]),
                    ("n_estimators", [150, 300, 500]), ("learning_rate", [0.02, 0.05, 0.1])]:
        if rng.random() < 0.3:
            d[k] = type(d[k])(pick(opts))
    for k in ("use_macro", "use_sentiment"):
        if rng.random() < 0.2:
            d[k] = not d[k]
    if rng.random() < 0.2:
        d["model_type"] = pick(["lgbm", "hgb", "rf"])
    return Genome(**_allowed(d, exclude_groups))


# ---------------------------------------------------------------- model
class SignalModel:
    """Regressor on forward returns -> continuous signal in [-1, +1] via tanh scaling."""

    def __init__(self, genome: Genome):
        self.genome = genome
        g = genome
        if g.model_type == "lgbm" and _HAS_LGBM:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=g.n_estimators, learning_rate=g.learning_rate,
                max_depth=g.max_depth, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=30, reg_lambda=1.0, random_state=7, n_jobs=-1, verbose=-1)
        elif g.model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=200, max_depth=g.max_depth + 2,
                                               min_samples_leaf=10, n_jobs=-1, random_state=7)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=g.n_estimators, learning_rate=g.learning_rate,
                max_depth=g.max_depth, l2_regularization=1.0, random_state=7)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.cols_ = list(X.columns)
        self.med_ = X.median(numeric_only=True)
        X = X.fillna(self.med_).fillna(0.0).clip(-8, 8)
        y = y.clip(y.quantile(0.02), y.quantile(0.98))
        self.model.fit(X, y)
        self.scale_ = float(np.std(self.model.predict(X))) + 1e-9
        return self

    def predict_signal(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols_).fillna(self.med_).fillna(0.0).clip(-8, 8)
        return np.tanh(self.model.predict(X) / (2.5 * self.scale_))

    def feature_importance(self) -> pd.Series:
        imp = getattr(self.model, "feature_importances_", None)
        return pd.Series(imp, index=self.cols_).sort_values() if imp is not None else pd.Series(dtype=float)


# ---------------------------------------------------------------- walk-forward
# A training row's label is the return over the NEXT `horizon` days. A row dated within `horizon`
# days of the test block therefore has a label that contains the test block's own returns — the
# model is trained on the answers. On pure random-walk data (nothing to predict) that alone gave
# an out-of-sample IC of +0.063 and a Sharpe of +0.33 (-0.002 and -0.63 once removed), and it
# rewards exactly what the lab then selects for: long horizons and many trees. Live trading never
# had this (labels not yet known are dropped), so the backtests flattered the strategy. Purging
# the last `horizon` days of every training window fixes it; universe.json ->
# evolution.purge_labels = false restores the old behaviour.
PURGE_LABELS = True


def configure_purge(flag=True) -> bool:
    global PURGE_LABELS
    PURGE_LABELS = bool(flag)
    return PURGE_LABELS


def _train_end(i: int, horizon: int) -> int:
    return max(0, i - horizon) if PURGE_LABELS else i


# What the model is asked to predict. "raw": the forward return, as before. "vol": that return
# divided by the asset's own recent volatility (vol_20 * sqrt(horizon)), i.e. "how many of its own
# standard deviations will it move". A pooled model trained on raw returns spends its capacity on
# whichever assets are most volatile (BTC moves ~3x AAPL, so it dominates a squared-error loss) and
# its signal strength isn't comparable across assets. On 20 real stocks (1997-2022, walk-forward,
# long-only) scaling the target lifted Sharpe by +0.25 (90% CI +0.11..+0.39); on 6 similar stocks
# it made no difference. universe.json -> evolution.target_scale = "raw" restores the old target.
TARGET_SCALE = "raw"


def configure_target(mode="raw") -> str:
    global TARGET_SCALE
    TARGET_SCALE = "vol" if str(mode).lower() in ("vol", "volatility", "scaled") else "raw"
    return TARGET_SCALE


def _target(d: pd.DataFrame, horizon: int) -> pd.Series:
    y = d[f"target_{horizon}"]
    if TARGET_SCALE == "vol":
        y = y / (d["vol_20"].clip(lower=0.003) * np.sqrt(horizon))
    return y.replace([np.inf, -np.inf], np.nan)


def _xy(d: pd.DataFrame, genome: "Genome", cols: list):
    """Training rows with a known label -> (features, target)."""
    y = _target(d, genome.horizon)
    ok = y.notna()
    return d.loc[ok, cols], y[ok]


def _block_bounds(dates, retrain_every: int) -> list:
    """[(first, end)) index ranges of the retrain blocks, on a CALENDAR grid: a block is every
    `retrain_every` calendar days (weekdays when the panel has no weekend rows, e.g. stocks only),
    counted from 1970. The old grid counted from the first row of whatever panel it was handed, so
    a score changed whenever the window moved by a single day — the retrain dates all shifted and
    every model was refit on a different sample (a champion's score swung by +-0.5 overnight with
    nothing new to learn). On this grid the signals for a past date don't change as data arrives."""
    d = pd.DatetimeIndex(dates)
    days = d.values.astype("datetime64[D]")
    if (d.dayofweek >= 5).any():
        n = days.astype(np.int64)
    else:
        n = np.busday_count(np.datetime64("1970-01-01"), days)
    bid = np.asarray(n) // max(1, int(retrain_every))
    cuts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
    return list(zip(cuts.tolist(), np.r_[cuts[1:], len(d)].tolist()))


def walk_forward_signals(panel: pd.DataFrame, genome: Genome, start=None):
    """Rolling retrain -> strictly out-of-sample daily signals (date x symbol).
    start: only produce signals from this date on (the training windows may reach back before it),
    so every genome can be judged on the same dates regardless of how much history it trains on."""
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    if len(dates) < genome.train_window + genome.horizon + 10:
        return pd.DataFrame(), None
    t0 = None if start is None else np.datetime64(pd.Timestamp(start))
    blocks, last_model = [], None
    for i, j in _block_bounds(dates, genome.retrain_every):
        if i < genome.train_window or (t0 is not None and dates[j - 1] < t0):
            continue
        tr = panel[panel["date"].isin(dates[i - genome.train_window:_train_end(i, genome.horizon)])]
        X, y = _xy(tr, genome, cols)
        te = panel[panel["date"].isin(dates[i:j])]
        if len(X) >= 200 and not te.empty:
            m = SignalModel(genome).fit(X, y)
            blocks.append(pd.DataFrame({"date": te["date"].values,
                                        "symbol": te["symbol"].values,
                                        "signal": m.predict_signal(te[cols])}))
            last_model = m
    if not blocks:
        return pd.DataFrame(), None
    sig = pd.concat(blocks).pivot_table(index="date", columns="symbol", values="signal").sort_index()
    if genome.smooth > 1:
        sig = sig.ewm(span=genome.smooth).mean()
    v = sig.to_numpy(dtype=float)  # linear dead-zone remap keeps continuity
    v = np.sign(v) * np.clip((np.abs(v) - genome.entry) / max(1e-9, 1 - genome.entry), 0, None)
    out = pd.DataFrame(v, index=sig.index, columns=sig.columns)
    return (out.loc[pd.Timestamp(start):] if start is not None else out), last_model


def diagnose_walk_forward(panel: pd.DataFrame, genome: Genome) -> pd.DataFrame:
    """Same window loop as walk_forward_signals, but reports per-window pass/skip instead of
    training anything — lets you see exactly which retrain windows produced no signal, and why,
    without re-running (and paying for) the real fit each time."""
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    rows = []
    if len(dates) < genome.train_window + genome.horizon + 10:
        return pd.DataFrame([{"test_start": None, "test_end": None, "train_rows": 0,
                              "status": "skipped", "reason": "not enough history at all "
                              f"({len(dates)} dates < train_window+horizon+10)"}])
    for i, j in _block_bounds(dates, genome.retrain_every):
        if i < genome.train_window:
            continue
        tr = panel[panel["date"].isin(dates[i - genome.train_window:_train_end(i, genome.horizon)])]
        X, _ = _xy(tr, genome, cols)
        te_dates = dates[i:j]
        te = panel[panel["date"].isin(te_dates)]
        ok = len(X) >= 200 and not te.empty
        reason = ("ok" if ok else
                 ("training rows below 200 after dropping NaN targets" if len(X) < 200
                  else "no test rows in this window (symbols missing data here?)"))
        rows.append({"test_start": pd.Timestamp(te_dates[0]) if len(te_dates) else None,
                     "test_end": pd.Timestamp(te_dates[-1]) if len(te_dates) else None,
                     "train_rows": len(X), "status": "ok" if ok else "skipped", "reason": reason})
    return pd.DataFrame(rows)


def train_final_model(panel: pd.DataFrame, genome: Genome) -> SignalModel:
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    d = panel[_target(panel, genome.horizon).notna()]
    dates = np.sort(d["date"].unique())[-genome.train_window:]
    X, y = _xy(d[d["date"].isin(dates)], genome, cols)
    return SignalModel(genome).fit(X, y)


def recent_signals(panel: pd.DataFrame, genome: Genome, model: SignalModel,
                   lookback: int = 90) -> pd.DataFrame:
    """Latest smoothed signals from a trained model (used by the headless pipeline)."""
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    dates = np.sort(panel["date"].unique())[-lookback:]
    recent = panel[panel["date"].isin(dates)]
    sig = recent.assign(s=model.predict_signal(recent[cols])) \
                .pivot_table(index="date", columns="symbol", values="s").sort_index()
    # Stocks have no bar on weekends/holidays, and crypto gets a bar for the new UTC day as soon as
    # it starts - so when a run lands after UTC midnight the panel's newest date is crypto-only and
    # every stock is NaN on the last row. EWM smoothing (smooth > 1) happens to carry values over
    # that gap, which is why this only broke once a champion with smooth=1 was promoted. Carry the
    # last real signal forward explicitly so it works for any genome; data_health() in the
    # pipeline still decides whether a stock is tradable on this session.
    sig = sig.ffill(limit=5)
    if genome.smooth > 1:
        sig = sig.ewm(span=genome.smooth).mean()
    v = sig.to_numpy(dtype=float)
    v = np.sign(v) * np.clip((np.abs(v) - genome.entry) / max(1e-9, 1 - genome.entry), 0, None)
    return pd.DataFrame(v, index=sig.index, columns=sig.columns)


# ---------------------------------------------------------------- evolution
def market_frames(panel: pd.DataFrame):
    """(closes, opens) date x symbol from a panel; opens is None for panels built without it."""
    pv = lambda col: panel.pivot_table(index="date", columns="symbol", values=col).sort_index()
    return pv("close").ffill(), (pv("open") if "open" in panel else None)


def score_genome(panel: pd.DataFrame, g: Genome, cost_bps: float, exec_open=(), sizing="equal",
                 frames=None, allow_short=False, eval_start=None):
    """Walk-forward backtest of one genome -> (fitness, metrics). The one evaluation path used
    for challengers AND the sitting champion.
    allow_short: score the book the account can actually hold. (This used to be left at
    run_backtest's default of True, so the lab picked genomes for a long/short book while the
    live account — allow_short false — only ever traded the long side.)
    eval_start: judge on the dates from here on, with models trained on the history before it
    (`panel` should reach back far enough), so genomes with different training windows are scored
    over exactly the same days and none of them is diluted by a flat warm-up."""
    prices, opens = frames or market_frames(panel)
    sig, _ = walk_forward_signals(panel, g, start=eval_start)
    if sig.empty:
        raise ValueError("no signals")
    m = run_backtest(prices, sig, fee_bps=cost_bps / 2, slippage_bps=cost_bps / 2,
                     max_leverage=g.max_leverage, allow_short=allow_short, opens=opens,
                     exec_open=exec_open, sizing=sizing).metrics
    return fitness(m), m


def hold_baseline(panel: pd.DataFrame, cost_bps: float, exec_open=(), sizing="equal",
                  allow_short=False, max_leverage=1.0, frames=None, eval_start=None):
    """The yardstick every strategy has to beat: hold EVERY asset long all the time, sized, costed
    and executed exactly like the strategy (same sizing string — so the same volatility target —
    same fees, same next-open fills), over the same dates -> (fitness, metrics). max_leverage is
    the gross-exposure cap — pass the strategy's own, so both are held to the same risk limit.
    On 20 real stocks (1997-2022) the walk-forward models did not beat this, so a model that can't
    is just adding cost and risk of being wrong; the lab and the backtest report it next to
    every strategy."""
    prices, opens = frames or market_frames(panel)
    sig = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    if eval_start is not None:
        sig = sig.loc[pd.Timestamp(eval_start):]
    m = run_backtest(prices, sig, fee_bps=cost_bps / 2, slippage_bps=cost_bps / 2,
                     max_leverage=max_leverage, allow_short=allow_short, opens=opens,
                     exec_open=exec_open, sizing=sizing).metrics
    return fitness(m), m


def sentiment_twin(g: Genome) -> Genome:
    """The same strategy with news sentiment switched the other way (model, features and
    hyperparameters untouched) — the one-variable comparison for "does sentiment help?"."""
    stem = g.name[:-5] if g.name.endswith(("+sent", "-sent")) else g.name
    return replace(g, use_sentiment=not g.use_sentiment,
                   name=f"{stem}{'-sent' if g.use_sentiment else '+sent'}")


def sentiment_information(panel: pd.DataFrame, g: Genome, alpha=300.0, block=10) -> tuple:
    """Does news sentiment add predictive information BEYOND the features this genome already
    uses? Two ridge regressions on next-day vol-standardized returns, walk-forward with the
    genome's own windows — one on the genome's features, one on those plus the sentiment
    features — and a paired comparison of what each earns out of sample (prediction x realized
    return, averaged over symbols per day, twin minus base, in blocks of `block` days, t-tested).
    Returns (t, n_blocks, mean daily difference); positive t = sentiment helps.

    Why linear, why one-day returns: comparing two tree ensembles' backtests is dominated by
    model-to-model noise (two near-identical models differ by +-0.5 fitness on luck alone), and
    10-day targets overlap, throwing away most of the independent data. A ridge pair is stable,
    and next-day returns give one independent observation per day. Measured on synthetic worlds
    (news mood either unrelated to returns or carrying a real edge): flagged pure noise as
    helpful in 0 of 100 runs; caught a modest real edge in roughly a third of runs and a
    stronger one in ~80%, vs ~7% for comparing the two tree backtests. A modest edge simply
    can't be proven from ~3 years of 6 tickers — hence the live shadow test too. Only dates
    from the first day the news store has coverage are used (before that the sentiment columns
    are constant and carry no information)."""
    from sklearn.linear_model import Ridge
    from .features import SENT_FEATURES
    base_cols = [c for c in feature_columns(replace(g, use_sentiment=False)) if c in panel.columns]
    sent_cols = [c for c in SENT_FEATURES if c in panel.columns]
    if not sent_cols or "news_cnt" not in panel.columns:
        raise ValueError("panel has no sentiment features")
    covered = panel.loc[panel["news_cnt"] > 0, "date"]
    if covered.empty:
        raise ValueError("no news coverage in the panel")
    p = panel[panel["date"] >= covered.min()].sort_values(["symbol", "date"]).copy()
    cl = p.set_index(["symbol", "date"])["close"]
    nxt = cl.groupby(level=0).shift(-1)
    p["y"] = ((nxt / cl - 1).to_numpy() / p["vol_20"].replace(0, np.nan).to_numpy()).clip(-6, 6)
    dates = np.sort(p["date"].unique())
    edge = {"base": {}, "twin": {}}
    for i in range(g.train_window, len(dates), g.retrain_every):
        tr = p[p["date"].isin(dates[i - g.train_window:i])].dropna(subset=["y"])
        te = p[p["date"].isin(dates[i:i + g.retrain_every])].dropna(subset=["y"])
        if len(tr) < 200 or te.empty:
            continue
        for tag, cols in (("base", base_cols), ("twin", base_cols + sent_cols)):
            med = tr[cols].median()
            Xtr = tr[cols].fillna(med).fillna(0.0)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
            Xtr = ((Xtr - mu) / sd).clip(-5, 5)
            Xte = ((te[cols].fillna(med).fillna(0.0) - mu) / sd).clip(-5, 5)
            pred = Ridge(alpha=alpha).fit(Xtr, tr["y"]).predict(Xte)
            e = pd.Series(pred * te["y"].to_numpy(), index=te["date"].to_numpy()).groupby(level=0).mean()
            edge[tag].update(e.to_dict())
    d = (pd.Series(edge["twin"]) - pd.Series(edge["base"])).dropna().sort_index()
    blocks = d.groupby(np.arange(len(d)) // block).mean()
    if len(d) % block < block // 2 and len(blocks) > 1:
        blocks = blocks.iloc[:-1]                         # drop a stub final block
    if len(blocks) < 6:
        return float("nan"), int(len(blocks)), float("nan")
    t = float(blocks.mean() / (blocks.std(ddof=1) / np.sqrt(len(blocks)) + 1e-12))
    return t, int(len(blocks)), float(d.mean())


def ablate_sentiment(panel: pd.DataFrame, g: Genome, cost_bps: float, exec_open=(), sizing="equal",
                     min_t=2.0, frames=None, info_panel=None, fitness_floor=-0.25,
                     allow_short=False, eval_start=None) -> dict:
    """Paired A/B of this genome vs the SAME genome with news sentiment switched the other way
    (same model, features, hyperparameters) — one planned comparison, not the best of many.

    Two separate questions, because a 2-year backtest can't answer the first one:
      1. Is there information?  sentiment_information(): a low-noise linear test on the longest
         history available (`info_panel`, default `panel`) -> info_t (positive = sentiment helps).
      2. Would trading it be sane?  The twin's walk-forward book (sized and costed like the real
         one, judged on the `panel` window) must not score clearly worse than the genome
         (fitness gain >= fitness_floor).
    `better` = the twin is the better variant: it needs |info_t| >= min_t in the right direction
    (sentiment on: info_t >= min_t; sentiment off: info_t <= -min_t) AND question 2."""
    prices, opens = frames or market_frames(panel)
    twin = sentiment_twin(g)
    runs = {}
    for tag, gg in (("base", g), ("twin", twin)):
        sig, _ = walk_forward_signals(panel, gg, start=eval_start)
        if sig.empty:
            raise ValueError("no signals")
        runs[tag] = sig
    idx = runs["base"].index.intersection(runs["twin"].index)
    if len(idx) < 60:
        raise ValueError("too little overlapping out-of-sample history for an A/B")
    mid = idx[len(idx) // 2]
    tgt = (panel.pivot_table(index="date", columns="symbol", values=f"target_{g.horizon}")
           .sort_index().reindex(idx))
    out = {}
    for tag, sig in runs.items():
        sig = sig.loc[idx]
        res = run_backtest(prices, sig, fee_bps=cost_bps / 2, slippage_bps=cost_bps / 2,
                           max_leverage=g.max_leverage, allow_short=allow_short, opens=opens,
                           exec_open=exec_open, sizing=sizing)
        full = res.metrics
        pair = pd.concat([sig.stack(), tgt.stack()], axis=1).dropna()
        out[tag] = {"full": fitness(full), "h1": fitness(window_metrics(res, None, mid)),
                    "h2": fitness(window_metrics(res, mid, None)),
                    "exposure": full["exposure"], "sharpe": full["sharpe"],
                    "max_drawdown": full["max_drawdown"], "cagr": full["cagr"],
                    "ic": float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
                    if len(pair) > 30 else float("nan")}
    info_t, n_blocks, _ = sentiment_information(info_panel if info_panel is not None else panel, g)
    info_t_twin = -info_t if g.use_sentiment else info_t          # positive = the twin is the better one
    delta = {k: out["twin"][k] - out["base"][k] for k in ("full", "h1", "h2")}
    return {"genome_uses_sentiment": bool(g.use_sentiment), "base": out["base"], "twin": out["twin"],
            "delta": delta, "info_t": info_t, "info_blocks": n_blocks, "min_t": min_t,
            "twin_genome": twin,
            "better": bool(info_t == info_t and info_t_twin >= min_t and delta["full"] >= fitness_floor)}


def evolve(panel: pd.DataFrame, pop_size=10, generations=4, seed=42,
           cost_bps=10.0, progress=None, exclude_groups=(), exec_open=(), sizing="equal",
           on_eval=None, allow_short=False, eval_start=None, time_budget=None):
    """Genetic search, ranked by backtest.fitness on a walk-forward backtest.
    exclude_groups: feature groups withheld from the search space (e.g. options/short/onchain
    while their real snapshot history is still too short to be meaningful).
    time_budget: seconds. Judging every genome over the same two years costs more than the old
    window did (a long-window genome used to be scored on a few months), and the daily workflow
    has a hard time limit; once the budget is spent the search stops after the generation in
    progress and ranks what it has instead of starting another."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    frames = market_frames(panel)
    cache: dict = {}

    def evaluate(g: Genome):
        if g.key() not in cache:
            try:
                cache[g.key()] = score_genome(panel, g, cost_bps, exec_open, sizing, frames,
                                              allow_short=allow_short, eval_start=eval_start)
            except Exception:
                cache[g.key()] = (-999.0, {})
        return cache[g.key()]

    def score_pop(pop, stage):
        """Scores a population one genome at a time. `progress` only fires once a whole
        generation is done - minutes into the run on a small server - so `on_eval(stage, stages,
        done, total, genome)` fires after EVERY candidate (cached elites count instantly). There
        are generations + 1 stages: the last one ranks the final children, which used to be
        scored silently after the bar had already reached 100%."""
        out = []
        for k, g in enumerate(pop, 1):
            out.append((g, evaluate(g)))
            if on_eval:
                on_eval(stage, generations + 1, k, len(pop), g)
        return out

    pop = [random_genome(rng, f"G0-{i}", exclude_groups) for i in range(pop_size)]
    history, stage = [], generations
    for gen in range(generations):
        scored = sorted(score_pop(pop, gen), key=lambda t: t[1][0], reverse=True)
        history.append({"generation": gen, "best": scored[0][1][0],
                        "avg": float(np.mean([s[1][0] for s in scored]))})
        if progress:
            progress(gen, generations, scored[0][1][0], scored[0][0])
        if time_budget and time.time() - t_start > time_budget and gen < generations - 1:
            stage = gen                                  # out of time: rank this generation as the final one
            break
        elites = [g for g, _ in scored[: max(2, pop_size // 4)]]
        nxt = [Genome(**asdict(e)) for e in elites]
        while len(nxt) < pop_size:
            a, b = elites[int(rng.integers(len(elites)))], elites[int(rng.integers(len(elites)))]
            nxt.append(mutate(crossover(a, b, rng), rng, exclude_groups))
        for i, g in enumerate(nxt):
            g.name = f"G{gen + 1}-{i}"
        pop = nxt

    rows = []
    for g, (score, m) in sorted(score_pop(pop, stage), key=lambda t: t[1][0], reverse=True):
        rows.append({"genome": g, "name": g.name, "score": round(score, 3),
                     **{k: (round(m[k], 3) if m.get(k) is not None else None)
                        for k in ["sharpe", "sortino", "max_drawdown", "cagr", "hit_rate", "exposure"]},
                     "model": g.model_type, "horizon": g.horizon, "entry": g.entry,
                     "smooth": g.smooth, "train": g.train_window, "retrain": g.retrain_every,
                     "lev": g.max_leverage, "groups": "+".join(g.feature_groups),
                     "macro": g.use_macro, "sent": g.use_sentiment})
    return rows, pd.DataFrame(history)


# ---------------------------------------------------------------- persistence
def save_bundle(path: str, genome: Genome, model: SignalModel):
    import joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"genome": genome, "model": model}, path)


def load_bundle(path: str):
    import joblib
    return joblib.load(path) if os.path.exists(path) else None
