"""A deliberately simple stock policy, used to turn forecast accuracy into stockouts and idle stock.

Each period, stock is topped up to S = forecast + z * sigma, where sigma is the root-mean-square
error of that model's own earlier one-step forecasts. A better model has a smaller sigma, so it
holds less safety stock for the same service target.

This is a one-period (newsvendor-style) policy with no carry-over, no lead time and no order cost.
It is a way to compare models, not a model of a real supply chain. Sales are used as demand, so
true demand is understated in any period when stock ran out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.645


def simulate(bt: pd.DataFrame, z: float = Z_95, warmup: int = 6) -> pd.DataFrame:
    """Return one row per product, model and evaluated period."""
    s1 = bt[bt["step"] == 1].sort_values(["product", "model", "date"])
    parts = []
    for (p, m), g in s1.groupby(["product", "model"]):
        if len(g) <= warmup:
            continue
        a = g["actual"].to_numpy(float)
        f = g["forecast"].to_numpy(float)
        e = a - f
        sigma = np.array([np.sqrt(np.mean(e[:t] ** 2)) for t in range(warmup, len(g))])
        order_up_to = np.maximum(f[warmup:] + z * sigma, 0.0)
        d = a[warmup:]
        parts.append(pd.DataFrame({
            "product": p, "model": m, "date": g["date"].to_numpy()[warmup:],
            "demand": d, "order_up_to": order_up_to,
            "sold": np.minimum(d, order_up_to),
            "short": np.maximum(d - order_up_to, 0.0),
            "leftover": np.maximum(order_up_to - d, 0.0),
            "stockout": d > order_up_to,
        }))
    if not parts:
        return pd.DataFrame(columns=["product", "model", "date", "demand", "order_up_to",
                                     "sold", "short", "leftover", "stockout"])
    return pd.concat(parts, ignore_index=True)


def summarise(periods: pd.DataFrame, by=("product", "model")) -> pd.DataFrame:
    g = periods.groupby(list(by))
    out = g.agg(periods=("demand", "size"), demand=("demand", "sum"), sold=("sold", "sum"),
                leftover=("leftover", "sum"), stockout_rate=("stockout", "mean"),
                mean_order_up_to=("order_up_to", "mean")).reset_index()
    out["fill_rate"] = np.where(out["demand"] > 0, out["sold"] / out["demand"], np.nan)
    out["idle_stock_pct"] = np.where(out["demand"] > 0, out["leftover"] / out["demand"], np.nan)
    return out[list(by) + ["periods", "stockout_rate", "fill_rate", "idle_stock_pct", "mean_order_up_to"]]


def vs_baseline(pooled: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Compare each model with the baseline: stockout change in percentage points, idle stock in relative %."""
    base = pooled[pooled["model"] == baseline]
    if base.empty:
        return pooled
    b = base.iloc[0]
    out = pooled.copy()
    out["stockout_change_pts"] = (out["stockout_rate"] - b["stockout_rate"]) * 100
    out["fill_rate_change_pts"] = (out["fill_rate"] - b["fill_rate"]) * 100
    out["idle_stock_change_pct"] = (out["idle_stock_pct"] / b["idle_stock_pct"] - 1) * 100
    return out
