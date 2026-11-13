import numpy as np
import pandas as pd
import pytest

from src import inventory
from src.evaluate import backtest, bias, mase, wape
from src.load_data import read_sales, resample_sales
from src.models import MODEL_NAMES, SeasonalNaive, make_model


# ---------- metrics, checked by hand ----------
def test_wape_known_value():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)


def test_bias_sign():
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)   # over-forecast is positive
    assert bias([10, 10], [8, 12]) == pytest.approx(0.0)


def test_mase_known_value():
    # in-sample one-step naive error is 1, forecast MAE is 2 -> MASE 2
    assert mase([5, 5], [7, 3], [1, 2, 3, 4, 5, 6], season=1) == pytest.approx(2.0)


def test_wape_zero_actual_is_nan():
    assert np.isnan(wape([0, 0], [1, 1]))


# ---------- loading ----------
def test_wide_to_long_and_weekly_sum(tmp_path):
    csv = tmp_path / "wide.csv"
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-08"],
                  "A": [1, 2, 4], "B": [10, 20, 40]}).to_csv(csv, index=False)
    long = read_sales(csv, fmt="wide", date_col="date")
    assert set(long["product"]) == {"A", "B"} and len(long) == 6
    weekly = resample_sales(long, "W")
    a = weekly[weekly["product"] == "A"].sort_values("date")
    assert list(a["units"]) == [3.0, 4.0]


def test_missing_date_column_gives_helpful_error(tmp_path):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"when": ["2024-01-01"], "A": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Date column"):
        read_sales(csv, date_col="date")


# ---------- models ----------
def test_seasonal_naive_repeats_last_season():
    m = SeasonalNaive(3, "W").fit([1, 2, 3, 1, 2, 3], None)
    assert list(m.predict(4)) == [1, 2, 3, 1]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_return_h_nonnegative_finite_values(name):
    rng = np.random.default_rng(1)
    n, m = 160, 12
    t = np.arange(n)
    y = 100 + 0.2 * t + 20 * np.sin(2 * np.pi * t / m) + rng.normal(0, 3, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    mdl = make_model(name, m, "MS").fit(y, dates)
    pred = mdl.predict(6)
    assert pred.shape == (6,) and np.all(np.isfinite(pred)) and np.all(pred >= 0)


def test_short_series_falls_back_and_says_so():
    y = np.arange(1.0, 11.0)
    dates = pd.date_range("2024-01-07", periods=10, freq="W")
    mdl = make_model("gbm", 52, "W").fit(y, dates)
    pred = mdl.predict(3)
    assert mdl.fallback is True and len(pred) == 3


def test_backtest_refuses_too_short_series():
    y = pd.Series(np.arange(60.0), index=pd.date_range("2020-01-05", periods=60, freq="W"))
    with pytest.raises(ValueError, match="too short"):
        backtest(y, "P", ["seasonal_naive"], 52, "W", h=4, test_periods=20)


def test_backtest_never_peeks_at_the_future():
    # Forecast for a date must not change if later data change.
    n = 140
    idx = pd.date_range("2018-01-07", periods=n, freq="W")
    base = np.sin(np.arange(n) / 5) * 10 + 100
    y1 = pd.Series(base, index=idx)
    y2 = y1.copy()
    y2.iloc[-3:] += 1000  # change only the last three observations
    bt1 = backtest(y1, "P", ["seasonal_naive"], 52, "W", h=2, test_periods=30, step=5)
    bt2 = backtest(y2, "P", ["seasonal_naive"], 52, "W", h=2, test_periods=30, step=5)
    early = bt1["origin"] < idx[-6]
    assert np.allclose(bt1[early]["forecast"].to_numpy(), bt2[early]["forecast"].to_numpy())


# ---------- inventory, checked by hand ----------
def test_inventory_known_numbers():
    dates = pd.date_range("2024-01-07", periods=4, freq="W")
    bt = pd.DataFrame({"product": "P", "model": "m", "origin": dates, "step": 1, "date": dates,
                       "actual": [90.0, 110.0, 120.0, 80.0], "forecast": 100.0, "fallback": False})
    per = inventory.simulate(bt, z=0.0, warmup=1)   # order-up-to = forecast = 100
    s = inventory.summarise(per).iloc[0]
    assert s["periods"] == 3
    assert s["stockout_rate"] == pytest.approx(2 / 3)       # demand 110 and 120 exceed 100
    assert s["fill_rate"] == pytest.approx(280 / 310)       # sold 100 + 100 + 80
    assert s["idle_stock_pct"] == pytest.approx(20 / 310)   # leftover only in the 80 period


def test_higher_z_means_fewer_stockouts():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-07", periods=60, freq="W")
    bt = pd.DataFrame({"product": "P", "model": "m", "origin": dates, "step": 1, "date": dates,
                       "actual": 100 + rng.normal(0, 15, 60), "forecast": 100.0, "fallback": False})
    lo = inventory.summarise(inventory.simulate(bt, z=0.5)).iloc[0]["stockout_rate"]
    hi = inventory.summarise(inventory.simulate(bt, z=2.0)).iloc[0]["stockout_rate"]
    assert hi <= lo
