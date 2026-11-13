-- Example analysis queries. Each block starts with "-- name: <id>" so the tests can run them.
-- Practise explaining every one of these out loud; interviews ask for exactly this.

-- name: monthly_totals
-- Units per product per calendar month.
SELECT strftime('%Y-%m', date) AS month,
       product,
       SUM(units)              AS units
FROM sales
GROUP BY month, product
ORDER BY product, month;

-- name: yearly_growth
-- Year-on-year growth per product using LAG() over yearly totals.
WITH yearly AS (
    SELECT product, strftime('%Y', date) AS year, SUM(units) AS units
    FROM sales
    GROUP BY product, year
)
SELECT product,
       year,
       units,
       LAG(units) OVER (PARTITION BY product ORDER BY year)            AS prev_units,
       ROUND(100.0 * (units - LAG(units) OVER (PARTITION BY product ORDER BY year))
             / LAG(units) OVER (PARTITION BY product ORDER BY year), 1) AS growth_pct
FROM yearly
ORDER BY product, year;

-- name: rolling_average
-- 4-period moving average (current row plus the three before it).
SELECT product,
       date,
       units,
       AVG(units) OVER (PARTITION BY product ORDER BY date
                        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS moving_avg_4
FROM sales
ORDER BY product, date;

-- name: abc_xyz
-- ABC = share of total volume (A: first 80%, B: next 15%, C: rest).
-- XYZ = demand variability, squared coefficient of variation (X < 0.25, Y < 1, Z otherwise).
-- SQLite has no STDEV, so variance = E[x^2] - E[x]^2.
WITH stats AS (
    SELECT product,
           SUM(units)                                      AS volume,
           AVG(units)                                      AS mean_units,
           AVG(units * units) - AVG(units) * AVG(units)    AS variance
    FROM sales
    GROUP BY product
),
ranked AS (
    SELECT *,
           SUM(volume) OVER (ORDER BY volume DESC
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 1.0
           / SUM(volume) OVER ()                           AS cum_share
    FROM stats
)
SELECT product,
       volume,
       ROUND(cum_share, 3) AS cum_share,
       CASE WHEN cum_share <= 0.80 THEN 'A'
            WHEN cum_share <= 0.95 THEN 'B'
            ELSE 'C' END   AS abc_class,
       ROUND(variance / (mean_units * mean_units), 3) AS cv_squared,
       CASE WHEN variance / (mean_units * mean_units) < 0.25 THEN 'X'
            WHEN variance / (mean_units * mean_units) < 1.0  THEN 'Y'
            ELSE 'Z' END   AS xyz_class
FROM ranked
ORDER BY volume DESC;

-- name: top_periods
-- The three biggest periods for each product.
SELECT product, date, units, rnk
FROM (
    SELECT product, date, units,
           RANK() OVER (PARTITION BY product ORDER BY units DESC) AS rnk
    FROM sales
)
WHERE rnk <= 3
ORDER BY product, rnk;
