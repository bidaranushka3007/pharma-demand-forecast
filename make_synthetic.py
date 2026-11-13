"""Generate SYNTHETIC weekly sales so the pipeline can be tested without real data.

This data is invented. Never report results from it as if they were real.
Usage: python scripts/make_synthetic.py [output.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_weeks=260, n_products=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-06", periods=n_weeks, freq="W")
    t = np.arange(n_weeks)
    cols = {}
    for i in range(n_products):
        level = rng.uniform(50, 400)
        trend = rng.uniform(-0.05, 0.25) * level / 52
        amp = rng.uniform(0.1, 0.4) * level
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.uniform(0.05, 0.25) * level
        y = level + trend * t + amp * np.sin(2 * np.pi * t / 52 + phase) + rng.normal(0, noise, n_weeks)
        spikes = rng.random(n_weeks) < 0.02
        y = y + spikes * level * rng.uniform(0.3, 0.8)
        cols[f"SYN{i + 1:02d}"] = np.maximum(np.round(y), 0)
    df = pd.DataFrame(cols)
    df.insert(0, "date", dates)
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/synthetic_sales_weekly.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)   # works even if data/raw was not uploaded
    make().to_csv(out, index=False)
    print("wrote", out, "(SYNTHETIC data, for testing only)")
