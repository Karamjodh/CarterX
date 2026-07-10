"""
CarterX — Forecasting Engine
─────────────────────────────────────────────────────────────────────────────
Generates revenue forecasts using LSTM or Prophet fallback.

STRATEGY:
  1. Build monthly revenue series from preprocessed data
  2. If >= 12 months of data → LSTM (Keras sequential model)
  3. If < 12 months          → Prophet (handles short series better)
  4. Output: daily forecast points for 30/60/90/180 days ahead
             with upper/lower confidence bands

OUTPUT CONTRACT (what ForecastTab.js reads):
  {
    "model_used":    "LSTM" | "Prophet",
    "history":       [{"date": "2023-01", "revenue": 12000}, ...],
    "forecast":      [{"date": "2024-02-01", "predicted": 13200,
                       "lower": 11800, "upper": 14600}, ...],
    "horizons":      [30, 60, 90, 180],
    "mae":           450.2,      # mean absolute error on validation set
    "has_date_data": true,
    "warning":       null | "string"
  }
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Confidence interval width (fraction of predicted value) ───────────────────
CI_WIDTH = 0.12   # ±12% → gives a visible but not absurd confidence band


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    success:      bool
    model_used:   str = ""
    history:      list = field(default_factory=list)   # monthly actuals
    forecast:     list = field(default_factory=list)   # daily forecast points
    horizons:     list = field(default_factory=lambda: [30, 60, 90, 180])
    mae:          float = 0.0
    has_date_data: bool = True
    warning:      Optional[str] = None
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point — called by pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def run_forecasting(df_clean: pd.DataFrame, dataset_type: str) -> ForecastResult:
    """
    Main entry point. Takes the cleaned DataFrame from preprocessing
    and returns a ForecastResult.

    Args:
        df_clean:     prep.df_clean (full cleaned dataset)
        dataset_type: "transactional" | "review" | "catalog"
    """
    try:
        # Build monthly revenue series
        monthly = _build_monthly_series(df_clean, dataset_type)

        if monthly is None or len(monthly) < 3:
            return ForecastResult(
                success      = True,
                model_used   = "none",
                has_date_data = False,
                warning      = (
                    "Not enough date-based data to generate a forecast. "
                    "Upload transactional data with dates to enable this feature."
                ),
            )

        history = [
            {"date": str(row["month"]), "revenue": round(float(row["revenue"]), 2)}
            for _, row in monthly.iterrows()
        ]

        # Choose model based on data length
        if len(monthly) >= 12:
            forecast, mae, model_name = _run_lstm(monthly)
        else:
            forecast, mae, model_name = _run_prophet(monthly)

        return ForecastResult(
            success      = True,
            model_used   = model_name,
            history      = history,
            forecast     = forecast,
            horizons     = [30, 60, 90, 180],
            mae          = round(float(mae), 2),
            has_date_data = True,
        )

    except Exception as e:
        logger.exception("Forecasting failed")
        return ForecastResult(
            success = False,
            error   = str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Monthly series builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly_series(df: pd.DataFrame, dataset_type: str) -> Optional[pd.DataFrame]:
    """Aggregates daily/transactional data into a monthly revenue series."""
    date_col = None
    for col in ["date", "order_date", "purchase_date", "invoice_date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        return None

    revenue_col = "revenue" if "revenue" in df.columns else (
        "price" if "price" in df.columns else None
    )

    if revenue_col is None:
        return None

    df["month"] = df[date_col].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month")[revenue_col]
        .sum()
        .reset_index()
        .rename(columns={revenue_col: "revenue"})
        .sort_values("month")
    )

    return monthly if len(monthly) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM forecaster
# ─────────────────────────────────────────────────────────────────────────────

def _run_lstm(monthly: pd.DataFrame) -> tuple[list, float, str]:
    """
    Trains a lightweight LSTM on the monthly revenue series.

    Architecture:
      Input → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)

    Training:
      - Sliding window of 3 months to predict next month
      - 80/20 train/validation split
      - Early stopping on val_loss (patience=10)
      - Max 100 epochs

    Returns (forecast_points, mae, model_name)
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel("ERROR")

    values = monthly["revenue"].values.astype(float)
    n      = len(values)

    # Normalise to [0, 1]
    v_min, v_max = values.min(), values.max()
    scale        = v_max - v_min if v_max > v_min else 1.0
    values_norm  = (values - v_min) / scale

    # Build sliding window sequences
    WINDOW = min(3, n - 1)

    X, y = [], []
    for i in range(len(values_norm) - WINDOW):
        X.append(values_norm[i : i + WINDOW])
        y.append(values_norm[i + WINDOW])

    X = np.array(X).reshape(-1, WINDOW, 1)
    y = np.array(y)

    split   = max(1, int(len(X) * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Model
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(WINDOW, 1), return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor  = "val_loss" if len(X_val) > 0 else "loss",
            patience = 10,
            restore_best_weights = True,
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val) if len(X_val) > 0 else None,
        epochs          = 100,
        batch_size      = max(1, min(8, len(X_train))),
        callbacks       = callbacks,
        verbose         = 0,
    )

    # MAE on validation set (or train if val too small)
    if len(X_val) > 0:
        val_preds = model.predict(X_val, verbose=0).flatten()
        mae_norm  = float(np.mean(np.abs(val_preds - y_val)))
    else:
        train_preds = model.predict(X_train, verbose=0).flatten()
        mae_norm    = float(np.mean(np.abs(train_preds - y_train)))
    mae = mae_norm * scale

    # ── Forecast 180 days ahead ────────────────────────────────────────────
    # We predict month-by-month, then interpolate to daily points
    last_window = values_norm[-WINDOW:].tolist()
    monthly_preds_norm = []

    for _ in range(6):   # 6 months ahead covers 180 days
        x_in  = np.array(last_window[-WINDOW:]).reshape(1, WINDOW, 1)
        pred  = float(model.predict(x_in, verbose=0)[0][0])
        monthly_preds_norm.append(pred)
        last_window.append(pred)

    # Denormalise
    monthly_preds = [p * scale + v_min for p in monthly_preds_norm]

    # Interpolate monthly predictions to daily points
    last_date   = pd.Period(monthly["month"].iloc[-1], "M")
    forecast    = _monthly_to_daily(monthly_preds, last_date)

    return forecast, mae, "LSTM"


# ─────────────────────────────────────────────────────────────────────────────
#  Prophet fallback
# ─────────────────────────────────────────────────────────────────────────────

def _run_prophet(monthly: pd.DataFrame) -> tuple[list, float, str]:
    """
    Uses Facebook Prophet for short series (< 12 months).
    Prophet handles seasonality and trend on limited data better than LSTM.
    """
    try:
        from prophet import Prophet
    except ImportError:
        # If prophet not installed, use simple linear extrapolation
        return _run_linear_fallback(monthly)

    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(monthly["month"]),
        "y":  monthly["revenue"].values,
    })

    model = Prophet(
        yearly_seasonality  = False,   # not enough data for yearly
        weekly_seasonality  = False,
        daily_seasonality   = False,
        interval_width      = 0.80,
        changepoint_prior_scale = 0.05,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    future   = model.make_future_dataframe(periods=180, freq="D")
    forecast  = model.predict(future)

    # MAE on historical portion
    hist_preds = forecast[forecast["ds"].isin(df_prophet["ds"])]["yhat"].values
    mae        = float(np.mean(np.abs(hist_preds - df_prophet["y"].values)))

    # Extract future portion only
    future_fc  = forecast[forecast["ds"] > df_prophet["ds"].max()].copy()
    future_fc["yhat"]        = future_fc["yhat"].clip(lower=0)
    future_fc["yhat_lower"]  = future_fc["yhat_lower"].clip(lower=0)
    future_fc["yhat_upper"]  = future_fc["yhat_upper"].clip(lower=0)

    points = [
        {
            "date":      row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(float(row["yhat"]), 2),
            "lower":     round(float(row["yhat_lower"]), 2),
            "upper":     round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_fc.iterrows()
    ]

    return points, mae, "Prophet"


# ─────────────────────────────────────────────────────────────────────────────
#  Linear extrapolation (last resort if Prophet not installed)
# ─────────────────────────────────────────────────────────────────────────────

def _run_linear_fallback(monthly: pd.DataFrame) -> tuple[list, float, str]:
    """Simple linear trend extrapolation when no ML library is available."""
    values = monthly["revenue"].values.astype(float)
    x      = np.arange(len(values))

    # Fit linear trend
    coeffs = np.polyfit(x, values, 1)
    trend  = np.poly1d(coeffs)

    # MAE
    mae = float(np.mean(np.abs(trend(x) - values)))

    # Forecast 180 days ahead (daily)
    last_date = pd.Period(monthly["month"].iloc[-1], "M")
    n_months  = 6
    monthly_preds = []
    for i in range(1, n_months + 1):
        monthly_preds.append(max(0, float(trend(len(values) - 1 + i))))

    points = _monthly_to_daily(monthly_preds, last_date)
    return points, mae, "Linear Trend"


# ─────────────────────────────────────────────────────────────────────────────
#  Monthly → Daily interpolation helper
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_to_daily(monthly_preds: list, last_history_month: pd.Period) -> list:
    """
    Converts monthly predicted values to daily forecast points
    by linear interpolation between month midpoints.
    Also adds ±CI_WIDTH confidence band around each prediction.
    """
    points = []
    current_month = last_history_month + 1

    for i, monthly_val in enumerate(monthly_preds):
        month_period = current_month + i
        # Get all days in this month
        start = month_period.to_timestamp("D", "S")
        end   = month_period.to_timestamp("D", "E")
        days  = pd.date_range(start, end, freq="D")

        # If we have next month's prediction, interpolate
        if i + 1 < len(monthly_preds):
            next_val = monthly_preds[i + 1]
            day_vals = np.linspace(monthly_val, next_val, len(days))
        else:
            day_vals = np.full(len(days), monthly_val)

        for day, val in zip(days, day_vals):
            val = max(0, float(val))
            points.append({
                "date":      day.strftime("%Y-%m-%d"),
                "predicted": round(val, 2),
                "lower":     round(val * (1 - CI_WIDTH), 2),
                "upper":     round(val * (1 + CI_WIDTH), 2),
            })

    return points