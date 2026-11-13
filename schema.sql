-- Tables used by the pipeline. SQLite syntax (window functions need SQLite 3.25+).

CREATE TABLE IF NOT EXISTS sales (
    date    TEXT NOT NULL,      -- ISO date, start or end of the period (see --freq)
    product TEXT NOT NULL,      -- product or drug-class code
    units   REAL NOT NULL,      -- units sold in the period
    PRIMARY KEY (product, date)
);

CREATE INDEX IF NOT EXISTS idx_sales_date ON sales (date);

-- Filled by the pipeline after the backtest.
CREATE TABLE IF NOT EXISTS accuracy (
    product TEXT, model TEXT, wape REAL, wape_step1 REAL, bias REAL,
    mase REAL, folds INTEGER, fallback_share REAL
);

CREATE TABLE IF NOT EXISTS inventory (
    product TEXT, model TEXT, periods INTEGER, stockout_rate REAL,
    fill_rate REAL, idle_stock_pct REAL, mean_order_up_to REAL
);
