from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import run_backtest
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


def random_genome(rng, name="") -> Genome:
    groups = [g for g in FEATURE_GROUPS if rng.random() < 0.6] or ["momentum"]
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    return Genome(name=name, model_type=pick(["lgbm", "lgbm", "hgb", "rf"]),
                  feature_groups=tuple(groups),
                  use_macro=bool(rng.random() < 0.7),
                  use_sentiment=bool(rng.random() < 0.6),
                  horizon=int(pick([3, 5, 10])), train_window=int(pick([252, 378, 504])),
                  retrain_every=int(pick([10, 21, 42])), smooth=int(pick([1, 2, 3, 5, 8])),
                  entry=float(np.round(rng.uniform(0.05, 0.35), 3)),
                  max_leverage=float(pick([0.5, 1.0, 1.5])),
                  n_estimators=int(pick([150, 300])),
                  learning_rate=float(pick([0.02, 0.05, 0.1])),
                  max_depth=int(pick([3, 4, 5, 6])))


def crossover(a: Genome, b: Genome, rng) -> Genome:
    da, db = asdict(a), asdict(b)
    d = {k: (da[k] if rng.random() < 0.5 else db[k]) for k in da if k != "name"}
    d["feature_groups"] = tuple(d["feature_groups"]) or ("momentum",)
    return Genome(name="", **d)


def mutate(g: Genome, rng) -> Genome:
    d = asdict(g)
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    if rng.random() < 0.5:
        groups = list(d["feature_groups"])
        if rng.random() < 0.5 and len(groups) > 1:
            groups.pop(int(rng.integers(len(groups))))
        else:
            cand = [x for x in FEATURE_GROUPS if x not in groups]
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
    return Genome(**d)


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
def walk_forward_signals(panel: pd.DataFrame, genome: Genome):
    """Rolling retrain -> strictly out-of-sample daily signals (date x symbol)."""
    tcol = f"target_{genome.horizon}"
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    if len(dates) < genome.train_window + genome.horizon + 10:
        return pd.DataFrame(), None
    blocks, last_model, i = [], None, genome.train_window
    while i < len(dates):
        tr = panel[panel["date"].isin(dates[i - genome.train_window:i])].dropna(subset=[tcol])
        te = panel[panel["date"].isin(dates[i:i + genome.retrain_every])]
        if len(tr) >= 200 and not te.empty:
            m = SignalModel(genome).fit(tr[cols], tr[tcol])
            blocks.append(pd.DataFrame({"date": te["date"].values,
                                        "symbol": te["symbol"].values,
                                        "signal": m.predict_signal(te[cols])}))
            last_model = m
        i += genome.retrain_every
    if not blocks:
        return pd.DataFrame(), None
    sig = pd.concat(blocks).pivot_table(index="date", columns="symbol", values="signal").sort_index()
    if genome.smooth > 1:
        sig = sig.ewm(span=genome.smooth).mean()
    v = sig.to_numpy(dtype=float)  # linear dead-zone remap keeps continuity
    v = np.sign(v) * np.clip((np.abs(v) - genome.entry) / max(1e-9, 1 - genome.entry), 0, None)
    return pd.DataFrame(v, index=sig.index, columns=sig.columns), last_model


def train_final_model(panel: pd.DataFrame, genome: Genome) -> SignalModel:
    tcol = f"target_{genome.horizon}"
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    d = panel.dropna(subset=[tcol])
    dates = np.sort(d["date"].unique())[-genome.train_window:]
    tr = d[d["date"].isin(dates)]
    return SignalModel(genome).fit(tr[cols], tr[tcol])


def recent_signals(panel: pd.DataFrame, genome: Genome, model: SignalModel,
                   lookback: int = 90) -> pd.DataFrame:
    """Latest smoothed signals from a trained model (used by the headless pipeline)."""
    cols = [c for c in feature_columns(genome) if c in panel.columns]
    dates = np.sort(panel["date"].unique())[-lookback:]
    recent = panel[panel["date"].isin(dates)]
    sig = recent.assign(s=model.predict_signal(recent[cols])) \
                .pivot_table(index="date", columns="symbol", values="s").sort_index()
    if genome.smooth > 1:
        sig = sig.ewm(span=genome.smooth).mean()
    v = sig.to_numpy(dtype=float)
    v = np.sign(v) * np.clip((np.abs(v) - genome.entry) / max(1e-9, 1 - genome.entry), 0, None)
    return pd.DataFrame(v, index=sig.index, columns=sig.columns)


# ---------------------------------------------------------------- evolution
def evolve(panel: pd.DataFrame, pop_size=10, generations=4, seed=42,
           cost_bps=10.0, progress=None):
    """Genetic search. Fitness = Sharpe - 0.5*|MaxDD| on walk-forward backtest."""
    rng = np.random.default_rng(seed)
    prices = panel.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    cache: dict = {}

    def evaluate(g: Genome):
        if g.key() not in cache:
            try:
                sig, _ = walk_forward_signals(panel, g)
                if sig.empty:
                    raise ValueError("no signals")
                m = run_backtest(prices, sig, fee_bps=cost_bps / 2,
                                 slippage_bps=cost_bps / 2, max_leverage=g.max_leverage).metrics
                score = m["sharpe"] - 0.5 * abs(m["max_drawdown"])
                if m["exposure"] < 0.05:
                    score -= 0.5
                cache[g.key()] = (float(score), m)
            except Exception:
                cache[g.key()] = (-999.0, {})
        return cache[g.key()]

    pop = [random_genome(rng, f"G0-{i}") for i in range(pop_size)]
    history = []
    for gen in range(generations):
        scored = sorted(((g, evaluate(g)) for g in pop), key=lambda t: t[1][0], reverse=True)
        history.append({"generation": gen, "best": scored[0][1][0],
                        "avg": float(np.mean([s[1][0] for s in scored]))})
        if progress:
            progress(gen, generations, scored[0][1][0], scored[0][0])
        elites = [g for g, _ in scored[: max(2, pop_size // 4)]]
        nxt = [Genome(**asdict(e)) for e in elites]
        while len(nxt) < pop_size:
            a, b = elites[int(rng.integers(len(elites)))], elites[int(rng.integers(len(elites)))]
            nxt.append(mutate(crossover(a, b, rng), rng))
        for i, g in enumerate(nxt):
            g.name = f"G{gen + 1}-{i}"
        pop = nxt

    rows = []
    for g, (score, m) in sorted(((g, evaluate(g)) for g in pop), key=lambda t: t[1][0], reverse=True):
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
