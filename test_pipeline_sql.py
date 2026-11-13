from pathlib import Path

import pandas as pd
import pytest

from scripts.make_synthetic import make
from src.load_data import run_sql_file
from src.run_pipeline import main

SQL = Path(__file__).resolve().parent.parent / "sql" / "queries.sql"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "syn.csv"
    make(n_weeks=170, n_products=3, seed=1).to_csv(csv, index=False)
    res = main(["--input", str(csv), "--date-col", "date", "--freq", "W", "--horizon", "4",
                "--test-periods", "30", "--step", "2", "--models", "seasonal_naive,ets,arima,gbm",
                "--db", str(d / "t.db"), "--out", str(d / "out")])
    return d, res


def test_pipeline_writes_all_outputs(run):
    d, res = run
    for f in ["backtest_forecasts.csv", "accuracy_summary.csv", "best_model_per_product.csv",
              "inventory_by_product.csv", "inventory_pooled.csv", "final_forecast.csv"]:
        assert (d / "out" / f).exists(), f
    fc = pd.read_csv(d / "out" / "final_forecast.csv")
    assert len(fc) == 3 * 4 and (fc["forecast"] >= 0).all()


def test_every_model_is_scored_for_every_product(run):
    _, res = run
    acc = res["accuracy"]
    per_product = acc[acc["product"] != "ALL (pooled)"]
    assert per_product.groupby("product")["model"].nunique().eq(4).all()
    assert per_product["wape"].between(0, 5).all()


def test_inventory_periods_match_across_models(run):
    _, res = run
    inv = res["inventory_pooled"]
    assert inv["periods"].nunique() == 1   # same periods for every model, so the comparison is fair
    assert "stockout_change_pts" in inv.columns


@pytest.mark.parametrize("name", ["monthly_totals", "yearly_growth", "rolling_average", "abc_xyz", "top_periods"])
def test_sql_queries_run(run, name):
    d, _ = run
    df = run_sql_file(d / "t.db", SQL, name)
    assert len(df) > 0


def test_abc_classes_are_valid(run):
    d, _ = run
    df = run_sql_file(d / "t.db", SQL, "abc_xyz")
    assert set(df["abc_class"]) <= {"A", "B", "C"}
    assert set(df["xyz_class"]) <= {"X", "Y", "Z"}
    assert df["cum_share"].iloc[-1] == pytest.approx(1.0)
