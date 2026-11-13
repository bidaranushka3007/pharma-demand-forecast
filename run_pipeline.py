"""End-to-end run: CSV -> SQLite -> backtest -> accuracy -> inventory simulation -> files for a dashboard.

Example:
    python scripts/make_synthetic.py
    python -m src.run_pipeline --input data/raw/synthetic_sales_weekly.csv --date-col date --freq W
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from . import inventory
from .evaluate import accuracy_summary, backtest, pick_best
from .load_data import read_series, read_sales, resample_sales, write_db
from .models import MODEL_NAMES, future_index, make_model

DEFAULT_SEASON = {"D": 7, "W": 52, "MS": 12, "ME": 12, "M": 12}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV of sales")
    p.add_argument("--format", dest="fmt", choices=["wide", "long"], default="wide")
    p.add_argument("--date-col", default="date")
    p.add_argument("--product-col", default="product")
    p.add_argument("--value-col", default="units")
    p.add_argument("--dayfirst", action="store_true")
    p.add_argument("--freq", default="W", help="D, W or MS (month start)")
    p.add_argument("--season", type=int, default=None, help="Periods per year or cycle (default from --freq)")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--test-periods", type=int, default=52)
    p.add_argument("--step", type=int, default=1, help="Move the backtest origin by this many periods")
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--z", type=float, default=inventory.Z_95, help="Service-level factor (1.645 = 95%%)")
    p.add_argument("--db", default="data/sales.db")
    p.add_argument("--out", default="outputs")
    p.add_argument("--products", default=None, help="Comma-separated subset of products")
    p.add_argument("--mlflow", action="store_true", help="Log runs to MLflow (needs mlflow installed)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    season = a.season or DEFAULT_SEASON.get(a.freq)
    if season is None:
        raise SystemExit("Give --season for this --freq")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if "seasonal_naive" not in models:
        models.insert(0, "seasonal_naive")  # the baseline is always needed for comparison
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load into SQL and read back, so analysis really runs off the database.
    sales = resample_sales(read_sales(a.input, a.fmt, a.date_col, a.product_col, a.value_col, a.dayfirst), a.freq)
    write_db(sales, a.db)
    series = read_series(a.db, a.freq)
    if a.products:
        keep = {s.strip() for s in a.products.split(",")}
        series = {k: v for k, v in series.items() if k in keep}
    print(f"Loaded {len(series)} products, {sales['date'].nunique()} periods each ({a.freq}).")

    # 2. Backtest every model on every product.
    frames = []
    for product, y in series.items():
        print(f"Backtesting {product} ...", flush=True)
        frames.append(backtest(y, product, models, season, a.freq, a.horizon, a.test_periods, a.step))
    bt = pd.concat(frames, ignore_index=True)
    bt.to_csv(out_dir / "backtest_forecasts.csv", index=False)

    # 3. Accuracy.
    acc = accuracy_summary(bt, series, season, a.test_periods)
    acc.to_csv(out_dir / "accuracy_summary.csv", index=False)
    best = pick_best(acc)
    best.to_csv(out_dir / "best_model_per_product.csv", index=False)

    # 4. Inventory simulation, per product and pooled.
    periods = inventory.simulate(bt, z=a.z)
    inv = inventory.summarise(periods)
    pooled = inventory.vs_baseline(inventory.summarise(periods, by=("model",)))
    inv.to_csv(out_dir / "inventory_by_product.csv", index=False)
    pooled.to_csv(out_dir / "inventory_pooled.csv", index=False)

    # 5. Final forecast: refit the best model per product on all data.
    fut = []
    for _, r in best.iterrows():
        y = series[r["product"]]
        mdl = make_model(r["best_model"], season, a.freq).fit(y.to_numpy(float), y.index)
        pred = mdl.predict(a.horizon)
        for d, v in zip(future_index(y.index[-1], a.horizon, a.freq), pred):
            fut.append((r["product"], d, float(v), r["best_model"]))
    pd.DataFrame(fut, columns=["product", "date", "forecast", "model"]).to_csv(
        out_dir / "final_forecast.csv", index=False)

    # 6. Results back into SQL, so a BI tool can connect to one file.
    con = sqlite3.connect(a.db)
    try:
        con.execute("DELETE FROM accuracy")
        con.execute("DELETE FROM inventory")
        acc.reindex(columns=["product", "model", "wape", "wape_step1", "bias", "mase", "folds",
                             "fallback_share"]).to_sql("accuracy", con, if_exists="append", index=False)
        inv.reindex(columns=["product", "model", "periods", "stockout_rate", "fill_rate",
                             "idle_stock_pct", "mean_order_up_to"]).to_sql("inventory", con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()

    # 7. Print the headline.
    show = acc[acc["product"] == "ALL (pooled)"][["model", "wape", "wape_step1", "bias", "fallback_share"]]
    print("\nPooled accuracy (WAPE: lower is better; bias > 0 means over-forecast)")
    print(show.round(3).to_string(index=False))
    print("\nInventory simulation, pooled, against seasonal_naive")
    cols = ["model", "stockout_rate", "fill_rate", "idle_stock_pct", "stockout_change_pts",
            "idle_stock_change_pct"]
    print(pooled[[c for c in cols if c in pooled.columns]].round(3).to_string(index=False))
    print(f"\nFiles written to {out_dir}/ and tables to {a.db}")

    if a.mlflow:
        _log_mlflow(a, acc, pooled, out_dir)
    return {"accuracy": acc, "inventory_pooled": pooled, "best": best}


def _log_mlflow(a, acc, pooled, out_dir):
    try:
        import mlflow
    except ImportError as e:
        raise SystemExit("Install mlflow first: pip install mlflow") from e
    store = Path(a.out).resolve() / "mlflow.db"   # newer MLflow versions need a database backend
    mlflow.set_tracking_uri(f"sqlite:///{store}")
    mlflow.set_experiment("pharma-demand-forecast")
    for m, g in acc[acc["product"] == "ALL (pooled)"].groupby("model"):
        with mlflow.start_run(run_name=m):
            mlflow.log_params({"model": m, "freq": a.freq, "horizon": a.horizon,
                               "test_periods": a.test_periods, "z": a.z})
            r = g.iloc[0]
            mlflow.log_metrics({"wape": float(r["wape"]), "wape_step1": float(r["wape_step1"]),
                                "bias": float(r["bias"])})
            pr = pooled[pooled["model"] == m]
            if len(pr):
                mlflow.log_metrics({"stockout_rate": float(pr.iloc[0]["stockout_rate"]),
                                    "idle_stock_pct": float(pr.iloc[0]["idle_stock_pct"])})
    print(f"Logged to MLflow. View with: mlflow ui --backend-store-uri sqlite:///{store}")


if __name__ == "__main__":
    main()
