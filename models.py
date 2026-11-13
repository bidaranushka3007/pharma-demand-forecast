"""Forecasting models with one shared interface: fit(y, dates) then predict(h).

Every model falls back to the seasonal-naive forecast if it fails to fit, and records that in
`.fallback`, so a silent failure never makes a model look better than it is.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

MODEL_NAMES = ["seasonal_naive", "ets", "arima", "gbm"]


def future_index(last, h: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(last, periods=h + 1, freq=freq)[1:]


def _fourier(t: np.ndarray, m: int, k: int = 3) -> np.ndarray:
    cols = []
    for j in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * j * t / m))
        cols.append(np.cos(2 * np.pi * j * t / m))
    return np.column_stack(cols)


class SeasonalNaive:
    """Repeat the last full season. The baseline every other model must beat."""
    name = "seasonal_naive"

    def __init__(self, season: int, freq: str):
        self.m, self.freq = season, freq
        self.fallback = False

    def fit(self, y, dates):
        self.y = np.asarray(y, float)
        return self

    def predict(self, h: int) -> np.ndarray:
        m = self.m if len(self.y) >= self.m else 1
        last = self.y[-m:]
        return np.array([last[k % m] for k in range(h)])


class _Base:
    name = "base"

    def __init__(self, season: int, freq: str):
        self.m, self.freq = season, freq
        self.fallback = False

    def fit(self, y, dates):
        self.y = np.asarray(y, float)
        self.dates = pd.DatetimeIndex(dates)
        self._base = SeasonalNaive(self.m, self.freq).fit(self.y, self.dates)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fit()
        except Exception:
            self.fallback = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.fallback:
            return self._base.predict(h)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = np.asarray(self._predict(h), float)
            if out.shape != (h,) or not np.all(np.isfinite(out)):
                raise ValueError("bad forecast")
            return np.maximum(out, 0.0)
        except Exception:
            self.fallback = True
            return self._base.predict(h)


class ETS(_Base):
    """Holt-Winters with a damped additive trend and additive seasonality when there is enough data."""
    name = "ets"

    def _fit(self):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if (self.m > 1 and len(self.y) >= 2 * self.m) else None
        self.res = ExponentialSmoothing(
            self.y, trend="add", damped_trend=True, seasonal=seasonal,
            seasonal_periods=self.m if seasonal else None,
            initialization_method="estimated",
        ).fit()

    def _predict(self, h):
        return self.res.forecast(h)


class ARIMAModel(_Base):
    """ARIMA(1,1,1). Seasonal AR term for short seasons; Fourier terms for long ones (e.g. weekly data)."""
    name = "arima"

    def _fit(self):
        from statsmodels.tsa.arima.model import ARIMA
        n = len(self.y)
        self.use_fourier = self.m > 12
        if self.use_fourier:
            self.res = ARIMA(self.y, exog=_fourier(np.arange(n), self.m), order=(1, 1, 1)).fit()
        elif self.m > 1:
            self.res = ARIMA(self.y, order=(1, 1, 1), seasonal_order=(1, 0, 0, self.m)).fit()
        else:
            self.res = ARIMA(self.y, order=(1, 1, 1)).fit()

    def _predict(self, h):
        if self.use_fourier:
            n = len(self.y)
            return self.res.forecast(h, exog=_fourier(np.arange(n, n + h), self.m))
        return self.res.forecast(h)


class GBM(_Base):
    """Gradient boosting on lag features, recursive multi-step.

    Targets and lags are divided by the recent level (mean of the last season), so the trees
    learn shape, not size. This lets one model cope with a growing or shrinking series.
    """
    name = "gbm"

    def __init__(self, season, freq):
        super().__init__(season, freq)
        self.lags = sorted(set(list(range(1, min(season, 4) + 1)) + [season]))

    def _level(self, hist) -> float:
        lvl = float(np.mean(hist[-self.m:]))
        return lvl if lvl > 0 else 1.0

    def _row(self, hist, date) -> list[float]:
        lvl = self._level(hist)
        lags = [hist[-l] / lvl for l in self.lags]
        roll4 = float(np.mean(hist[-4:])) / lvl
        doy = date.dayofyear
        return lags + [roll4, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]

    def _fit(self):
        maxlag = max(self.lags)
        if len(self.y) < maxlag + 20:
            raise ValueError("series too short for gbm")
        X, t = [], []
        for i in range(maxlag, len(self.y)):
            hist = self.y[:i]
            X.append(self._row(hist, self.dates[i]))
            t.append(self.y[i] / self._level(hist))
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        self.model.fit(np.array(X), np.array(t))

    def _predict(self, h):
        hist = list(self.y)
        dates = future_index(self.dates[-1], h, self.freq)
        out = []
        for d in dates:
            lvl = self._level(hist)
            pred = float(self.model.predict(np.array([self._row(hist, d)]))[0]) * lvl
            pred = max(pred, 0.0)
            out.append(pred)
            hist.append(pred)
        return out


_REGISTRY = {"seasonal_naive": SeasonalNaive, "ets": ETS, "arima": ARIMAModel, "gbm": GBM}


def make_model(name: str, season: int, freq: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](season, freq)
