from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .features import FEATURE_GROUPS
from .models import Genome, market_frames, score_genome


def _suggest(trial, exclude_groups=()) -> Genome:
    eligible = [g for g in FEATURE_GROUPS if g not in exclude_groups]
    groups = [g for g in eligible if trial.suggest_categorical(f"grp_{g}", [0, 1])]
    return Genome(
        name=f"T{trial.number}", feature_groups=tuple(groups or ["momentum"]),
        model_type=trial.suggest_categorical("model_type", ["lgbm", "hgb", "rf"]),
        use_macro="macro" not in exclude_groups and bool(trial.suggest_categorical("use_macro", [0, 1])),
        use_sentiment=("sentiment" not in exclude_groups
                       and bool(trial.suggest_categorical("use_sentiment", [0, 1]))),
        horizon=trial.suggest_categorical("horizon", [3, 5, 10]),
        train_window=trial.suggest_categorical("train_window", [252, 378, 504]),
        retrain_every=trial.suggest_categorical("retrain_every", [10, 21, 42]),
        smooth=trial.suggest_categorical("smooth", [1, 2, 3, 5, 8]),
        entry=trial.suggest_float("entry", 0.01, 0.15, log=True),
        max_leverage=trial.suggest_categorical("max_leverage", [0.5, 1.0, 1.5]),
        n_estimators=trial.suggest_categorical("n_estimators", [150, 300, 500]),
        learning_rate=trial.suggest_categorical("learning_rate", [0.02, 0.05, 0.1]),
        max_depth=trial.suggest_categorical("max_depth", [3, 4, 5, 6]))


def optimize(panel, n_trials=40, seed=42, cost_bps=10.0, storage=None,
             study_name="adaptive-trader", timeout=None, progress=None, exclude_groups=(),
             exec_open=(), sizing="equal"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frames = market_frames(panel)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
                                storage=storage, study_name=study_name, load_if_exists=True)

    def objective(trial):
        g = _suggest(trial, exclude_groups)
        trial.set_user_attr("genome", asdict(g))
        try:
            score, m = score_genome(panel, g, cost_bps, exec_open, sizing, frames)
            for k in ("sharpe", "sortino", "max_drawdown", "cagr", "hit_rate", "exposure"):
                trial.set_user_attr(k, float(m[k]))
            return score
        except Exception:
            return -999.0

    cbs = [(lambda s, t: progress(s, t))] if progress else []
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=cbs, gc_after_trial=True)

    rows = []
    for t in sorted((t for t in study.trials if t.value is not None and t.value > -900),
                    key=lambda t: t.value, reverse=True):
        g = Genome(**t.user_attrs["genome"])
        rows.append({"genome": g, "name": g.name, "score": round(t.value, 3),
                     **{k: round(t.user_attrs.get(k, float("nan")), 3)
                        for k in ("sharpe", "sortino", "max_drawdown", "cagr", "hit_rate", "exposure")},
                     "model": g.model_type, "horizon": g.horizon, "entry": round(g.entry, 3),
                     "smooth": g.smooth, "train": g.train_window, "retrain": g.retrain_every,
                     "lev": g.max_leverage, "groups": "+".join(g.feature_groups),
                     "macro": g.use_macro, "sent": g.use_sentiment})
    vals = [t.value if t.value is not None else np.nan for t in study.trials]
    hist = pd.DataFrame({"generation": range(len(vals)), "best": pd.Series(vals).cummax(),
                         "avg": pd.Series(vals).rolling(5, min_periods=1).mean()})
    return rows, hist, study
