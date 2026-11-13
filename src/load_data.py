"""Read a sales CSV, regularise it to one frequency and store it in SQLite."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def read_sales(path, fmt="wide", date_col="date", product_col="product",
               value_col="units", dayfirst=False) -> pd.DataFrame:
    """Return a long table with columns date, product, units.

    fmt="wide": one date column and one column per product (common for pharma-class data).
    fmt="long": one row per date and product.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Columns are: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)

    if fmt == "wide":
        long = df.melt(id_vars=date_col, var_name="product", value_name="units")
        long = long.rename(columns={date_col: "date"})
    elif fmt == "long":
        for c in (product_col, value_col):
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found. Columns are: {list(df.columns)}")
        long = df.rename(columns={date_col: "date", product_col: "product",
                                  value_col: "units"})[["date", "product", "units"]]
    else:
        raise ValueError("fmt must be 'wide' or 'long'")

    long["product"] = long["product"].astype(str)
    long["units"] = pd.to_numeric(long["units"], errors="coerce")
    return long.dropna(subset=["date"]).reset_index(drop=True)


def resample_sales(long: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Sum units into regular periods. Missing periods become 0, so check your data first."""
    clean = long.dropna(subset=["units"])
    out = (clean.set_index("date")
                .groupby("product")["units"]
                .resample(freq).sum()
                .reset_index())
    return out[["date", "product", "units"]]


def write_db(sales: pd.DataFrame, db_path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript((SQL_DIR / "schema.sql").read_text())
        con.execute("DELETE FROM sales")
        rows = [(d.strftime("%Y-%m-%d"), p, float(u))
                for d, p, u in zip(sales["date"], sales["product"], sales["units"])]
        con.executemany("INSERT INTO sales (date, product, units) VALUES (?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()


def read_series(db_path, freq: str) -> dict[str, pd.Series]:
    """Read every product's series back out of SQL as a regular pandas Series."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT date, product, units FROM sales ORDER BY product, date", con)
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    series = {}
    for product, g in df.groupby("product"):
        s = g.set_index("date")["units"].astype(float).asfreq(freq)
        if s.isna().any():
            s = s.fillna(0.0)
        series[str(product)] = s
    return series


def run_sql_file(db_path, sql_path, name: str) -> pd.DataFrame:
    """Run one named block ('-- name: xyz') from a queries file and return the result."""
    text = Path(sql_path).read_text()
    blocks = {}
    current = None
    for line in text.splitlines():
        if line.startswith("-- name:"):
            current = line.split(":", 1)[1].strip()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    if name not in blocks:
        raise KeyError(f"No query named '{name}'. Available: {list(blocks)}")
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("\n".join(blocks[name]), con)
    finally:
        con.close()
