# Pharma demand forecasting and stock-risk pipeline

> **Status: code complete and tested on synthetic data. Real-data results: not yet run.**
> Replace this line, and fill the Results section, after you run it on a real pharma sales dataset.

## Problem

Pharma supply chains must hold enough stock to avoid shortages without tying up money in unsold
product. Both depend on how good the demand forecast is. This project forecasts weekly drug sales
with four methods, scores them honestly, and turns forecast error into two numbers a supply chain
team cares about: **stockout rate** and **idle stock**.

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_synthetic.py            # SYNTHETIC data, for testing only
python -m src.run_pipeline --input data/raw/synthetic_sales_weekly.csv --date-col date --freq W
pytest                                      # 25 tests
```

A full run on 6 products and 260 weeks takes about 2 minutes. For a quick trial add
`--products SYN01 --step 4`.

## Using real data

1. Download a pharma sales file into `data/raw/`. I believe there is a public Kaggle dataset of
   pharmacy sales by drug class (daily, weekly and monthly files, one column per drug class, with a
   date column named `datum`). **I have not verified this. Open the file and check the columns,
   the dates and the licence before you rely on it.**
2. Run (adjust the flags to what you see in the file):

```bash
python -m src.run_pipeline --input data/raw/salesweekly.csv --date-col datum --freq W --test-periods 52
```

Use `--format long --product-col ... --value-col ...` for one-row-per-sale files, `--freq MS` for
monthly data, `--dayfirst` if dates are day/month/year.

If the dates look wrong or periods are missing, fix that first. Missing periods are filled with 0,
which is wrong if the gap means "no record" rather than "no sales".

## What the pipeline does

| Step | File | What happens |
|---|---|---|
| 1. Load | `src/load_data.py`, `sql/schema.sql` | CSV is cleaned, summed into regular periods and stored in SQLite. Analysis reads back from SQL. |
| 2. SQL analysis | `sql/queries.sql` | Monthly totals, year-on-year growth (`LAG`), moving average, ABC/XYZ product classification, top periods. |
| 3. Forecast | `src/models.py` | Seasonal naive (baseline), exponential smoothing, ARIMA, gradient boosting. |
| 4. Backtest | `src/evaluate.py` | Rolling origin: refit at each step using only past data, forecast `h` periods ahead. |
| 5. Score | `src/evaluate.py` | WAPE, MASE, bias, per product and pooled. Fallbacks are counted, never hidden. |
| 6. Stock policy | `src/inventory.py` | Order-up-to level = forecast + z * past error. Counts stockouts, fill rate, idle stock. |
| 7. Output | `outputs/`, `data/sales.db` | CSVs and SQL tables for a Power BI or Tableau dashboard; optional MLflow logging (`--mlflow`). |

## Reading the metrics

* **WAPE**: total absolute error divided by total sales. 0.10 means forecasts are off by 10% of volume. Lower is better.
* **MASE**: error relative to the seasonal-naive baseline. Below 1 beats it.
* **Bias**: positive means over-forecasting (builds idle stock); negative means under-forecasting (causes stockouts).
* **Stockout rate**: share of periods where demand exceeded stock.
* **Idle stock**: leftover units as a share of demand.

## Results

*Fill this in from `outputs/accuracy_summary.csv` and `outputs/inventory_pooled.csv` after a real run.*

| Model | WAPE | Bias | Stockout rate | Idle stock | vs baseline |
|---|---|---|---|---|---|
| seasonal_naive | | | | | |
| ets | | | | | |
| arima | | | | | |
| gbm | | | | | |

Write two or three sentences: which model won, by how much, and **whether it beat the baseline at
all**. If it did not, say so. A clear negative result is better than a hidden one.

## Limitations (keep these; they are the honest part)

* The stock policy is a one-period simplification: no lead time, no order cost, no carry-over.
  It compares models; it does not model a real supply chain.
* Sales are used as demand. When stock ran out, true demand was higher than recorded sales.
* Results from the synthetic generator are meaningless. Never quote them.
* Public sales data may not resemble a manufacturer's own demand (distributor orders, tenders,
  seasonality). Say what your data is, and what it is not.
* ARIMA uses Fourier terms for yearly seasonality on weekly data; this is a simple choice, not a tuned one.

## Dashboard (Power BI or Tableau)

Import the CSVs in `outputs/` (or connect to `data/sales.db`). Suggested pages:

1. **Forecast vs actual** per product (`backtest_forecasts.csv`), with a product slicer.
2. **Model accuracy** (`accuracy_summary.csv`): WAPE by model, with the baseline highlighted.
3. **Stock risk** (`inventory_by_product.csv`): stockout rate against idle stock per model.
4. **Next 4 periods** (`final_forecast.csv`) and the ABC/XYZ table from `sql/queries.sql`.

## MLflow

`python -m src.run_pipeline ... --mlflow` logs one run per model to `outputs/mlflow.db`. View it with
`mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`. This needs the full `mlflow` package;
`mlflow-skinny` also needs `sqlalchemy` and `alembic`.

## Be ready to explain in an interview

* Why a rolling-origin backtest instead of one train/test split.
* Why the baseline matters, and what you would do if a model failed to beat it.
* Why WAPE and not MAPE (division by small values), and what bias tells a planner.
* What the inventory simulation leaves out.
* Every query in `sql/queries.sql`, line by line.
