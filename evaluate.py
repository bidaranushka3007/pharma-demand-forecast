"""Rolling-origin backtest and accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import make_model

BT_COLUMNS = ["product", "model", "origin", "step", "date", "actual", "forecast", "fallback"]


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: total absolute error / total actual. Lower is better."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - f)) / denom) if denom > 0 else float("nan")


def bias(actual, forecast) -> float:
    """Positive = forecasts run high (over-forecasting, which builds idle stock)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.sum(np.abs(a))
    return float(np.sum(f - a) / denom) if denom > 0 else float("nan")


def mase(actual, forecast, train, season: int) -> float:
    """Mean absolute error scaled by the in-sample seasonal-naive error. Below 1 beats the baseline."""
    a, f, tr = np.asarray(actual, float), np.asarray(forecast, float), np.asarray(train, float)
    m = season if len(tr) > season else 1
    scale = np.mean(np.abs(tr[m:] - tr[:-m])) if len(tr) > m else 0.0
    return float(np.mean(np.abs(a - f)) / scale) if scale > 0 else float("nan")


def backtest(y: pd.Series, product: str, model_names, season: int, freq: str,
             h: int, test_periods: int, step: int = 1) -> pd.DataFrame:
    """Refit each model at every origin and forecast h steps ahead, using only data before the origin."""
    n = len(y)
    first = n - test_periods
    min_train = max(2 * season, 30)
    if first < min_train:
        raise ValueError(
            f"{product}: {n} periods is too short. Need at least {min_train + test_periods} "
            f"(training {min_train} + test {test_periods}). Lower --test-periods or use more history.")
    vals, idx = y.to_numpy(float), y.index
    rows = []
    for o in range(first, n - h + 1, step):
        for name in model_names:
            mdl = make_model(name, season, freq).fit(vals[:o], idx[:o])
            pred = mdl.predict(h)
            for k in range(h):
                rows.append((product, name, idx[o - 1], k + 1, idx[o + k],
                             vals[o + k], float(pred[k]), bool(mdl.fallback)))
    return pd.DataFrame(rows, columns=BT_COLUMNS)


def accuracy_summary(bt: pd.DataFrame, series_map: dict, season: int, test_periods: int) -> pd.DataFrame:
    rows = []
    for (p, m), g in bt.groupby(["product", "model"]):
        y = series_map[p].to_numpy(float)
        train = y[: len(y) - test_periods]
        g1 = g[g["step"] == 1]
        rows.append(dict(
            product=p, model=m,
            wape=wape(g["actual"], g["forecast"]),
            wape_step1=wape(g1["actual"], g1["forecast"]),
            bias=bias(g["actual"], g["forecast"]),
            mase=mase(g["actual"], g["forecast"], train, season),
            folds=int(g["origin"].nunique()),
            fallback_share=float(g["fallback"].mean()),
        ))
    out = pd.DataFrame(rows)
    pooled = []
    for m, g in bt.groupby("model"):
        g1 = g[g["step"] == 1]
        pooled.append(dict(
            product="ALL (pooled)", model=m,
            wape=wape(g["actual"], g["forecast"]),
            wape_step1=wape(g1["actual"], g1["forecast"]),
            bias=bias(g["actual"], g["forecast"]),
            mase=float("nan"), folds=int(g["origin"].nunique()),
            fallback_share=float(g["fallback"].mean())))
    return pd.concat([out, pd.DataFrame(pooled)], ignore_index=True)


def pick_best(summary: pd.DataFrame) -> pd.DataFrame:
    """Best model per product by WAPE, with the baseline's WAPE alongside for honesty."""
    prods = summary[summary["product"] != "ALL (pooled)"]
    rows = []
    for p, g in prods.groupby("product"):
        g = g.dropna(subset=["wape"])
        if g.empty:
            continue
        best = g.loc[g["wape"].idxmin()]
        base = g[g["model"] == "seasonal_naive"]["wape"]
        rows.append(dict(product=p, best_model=best["model"], best_wape=best["wape"],
                         baseline_wape=float(base.iloc[0]) if len(base) else float("nan")))
    return pd.DataFrame(rows)
