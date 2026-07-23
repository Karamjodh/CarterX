# BEFORE
"""
...
STRATEGY:
  1. Build monthly revenue series from preprocessed data
  2. If >= 12 months of data → LSTM (Keras sequential model)
  3. If < 12 months          → Prophet (handles short series better)
  4. Output: daily forecast points for 30/60/90/180 days ahead
             with upper/lower confidence bands
...
"""

# AFTER
"""
...
STRATEGY:
  1. Build monthly revenue series from preprocessed data
  2. If >= 24 months of data → LSTM (Keras sequential model)
     LSTM needs at least 2 full years to learn seasonal patterns.
     With less data it collapses to predicting the mean.
  3. If 3–23 months          → Prophet (Facebook)
     Handles short series better — fits trend + seasonality directly
     without needing long sequences.
  4. If < 3 months           → No forecast (not enough signal)
  5. Output: monthly forecast points for 1/2/3/6 months ahead
             with ±12% confidence band
...
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

CI_WIDTH = 0.12   # ±12% confidence band


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    success:       bool
    model_used:    str  = ""
    history:       list = field(default_factory=list)
    forecast:      list = field(default_factory=list)
    horizons:      list = field(default_factory=lambda: [30, 60, 90, 180])
    mae:           float = 0.0
    has_date_data: bool  = True
    warning:       Optional[str] = None
    error:         Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_forecasting(df_clean: pd.DataFrame, dataset_type: str) -> ForecastResult:
    try:
        monthly = _build_monthly_series(df_clean, dataset_type)

        if monthly is None or len(monthly) < 3:
            return ForecastResult(
                success       = True,
                model_used    = "none",
                has_date_data = False,
                warning       = (
                    "Not enough date-based data to generate a forecast. "
                    "Upload transactional data with dates to enable this feature."
                ),
            )

        history = [
            {"date": str(row["month"]), "revenue": round(float(row["revenue"]), 2)}
            for _, row in monthly.iterrows()
        ]

        # Need at least 24 months for LSTM to learn meaningful patterns
        # UCI dataset has only 13 months → use Prophet which handles short series better
        if len(monthly) >= 24:
            forecast, mae, model_name = _run_lstm(monthly)
        else:
            forecast, mae, model_name = _run_prophet(monthly)

        return ForecastResult(
            success       = True,
            model_used    = model_name,
            history       = history,
            forecast      = forecast,
            horizons      = [1, 2, 3, 6],   # months ahead
            mae           = round(float(mae), 2),
            has_date_data = True,
        )

    except Exception as e:
        logger.exception("Forecasting failed")
        return ForecastResult(success=False, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Monthly series builder — FIXED
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly_series(df: pd.DataFrame, dataset_type: str) -> Optional[pd.DataFrame]:
    """
    Builds a monthly revenue series from the cleaned DataFrame.

    Revenue resolution order:
      1. "revenue" column  — set by _engineer_features (quantity × price)
      2. "total_sales" / "total_revenue" / "amount" — direct revenue columns
      3. price × quantity  — computed on the fly if both exist
      4. "price" alone     — last resort

    Date resolution order:
      Checks all known date column names after canonical mapping.
    """
    # ── Find date column ───────────────────────────────────────────────────
    date_col = None
    for col in ["date", "invoice_date", "order_date", "purchase_date",
                "created_at", "invoicedate", "sale_date", "order_timestamp"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning("Forecasting: no date column found in df_clean")
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        logger.warning("Forecasting: all date values were NaT after coercion")
        return None

    # ── Find or compute revenue ────────────────────────────────────────────
    revenue_col = None

    # Priority 1: already-computed revenue column (from _engineer_features)
    for col in ["revenue", "total_sales", "total_revenue",
                "amount", "total_amount", "net_sales"]:
        if col in df.columns:
            revenue_col = col
            logger.info("Forecasting: using revenue column '%s'", col)
            break

    # Priority 2: compute price × quantity on the fly
    if revenue_col is None:
        if "price" in df.columns and "quantity" in df.columns:
            df["_revenue"] = (
                pd.to_numeric(df["price"],    errors="coerce").fillna(0) *
                pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
            )
            revenue_col = "_revenue"
            logger.info("Forecasting: computed revenue = price × quantity")

        elif "price" in df.columns:
            revenue_col = "price"
            logger.info("Forecasting: using 'price' as revenue proxy")

    if revenue_col is None:
        logger.warning("Forecasting: no revenue column found")
        return None

    # ── Ensure numeric and positive ────────────────────────────────────────
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    df = df[df[revenue_col] > 0]

    if df.empty:
        logger.warning("Forecasting: no positive revenue rows after filtering")
        return None

    # ── Aggregate monthly ──────────────────────────────────────────────────
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month")[revenue_col]
        .sum()
        .reset_index()
        .rename(columns={revenue_col: "revenue"})
        .sort_values("month")
        .reset_index(drop=True)
    )

    logger.info(
        "Forecasting: %d monthly points | revenue range %.2f – %.2f",
        len(monthly), monthly["revenue"].min(), monthly["revenue"].max()
    )

    return monthly if len(monthly) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM forecaster
# ─────────────────────────────────────────────────────────────────────────────

def _run_lstm(monthly: pd.DataFrame) -> tuple[list, float, str]:
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel("ERROR")

    values = monthly["revenue"].values.astype(float)
    n      = len(values)

    v_min, v_max = values.min(), values.max()
    scale        = v_max - v_min if v_max > v_min else 1.0
    values_norm  = (values - v_min) / scale

    WINDOW = min(3, n - 1)

    X, y = [], []
    for i in range(len(values_norm) - WINDOW):
        X.append(values_norm[i : i + WINDOW])
        y.append(values_norm[i + WINDOW])

    X = np.array(X).reshape(-1, WINDOW, 1)
    y = np.array(y)

    split          = max(1, int(len(X) * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(WINDOW, 1), return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss" if len(X_val) > 0 else "loss",
            patience             = 10,
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

    if len(X_val) > 0:
        val_preds = model.predict(X_val, verbose=0).flatten()
        mae_norm  = float(np.mean(np.abs(val_preds - y_val)))
    else:
        train_preds = model.predict(X_train, verbose=0).flatten()
        mae_norm    = float(np.mean(np.abs(train_preds - y_train)))
    mae = mae_norm * scale

    last_window        = values_norm[-WINDOW:].tolist()
    monthly_preds_norm = []

    for _ in range(6):
        x_in = np.array(last_window[-WINDOW:]).reshape(1, WINDOW, 1)
        pred = float(model.predict(x_in, verbose=0)[0][0])
        monthly_preds_norm.append(pred)
        last_window.append(pred)

    monthly_preds = [p * scale + v_min for p in monthly_preds_norm]
    last_date     = pd.Period(monthly["month"].iloc[-1], "M")
    forecast      = _build_monthly_forecast(monthly_preds, last_date)

    return forecast, mae, "LSTM"


# ─────────────────────────────────────────────────────────────────────────────
#  Prophet fallback
# ─────────────────────────────────────────────────────────────────────────────

def _run_prophet(monthly: pd.DataFrame) -> tuple[list, float, str]:
    try:
        from prophet import Prophet
    except ImportError:
        return _run_linear_fallback(monthly)

    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(monthly["month"]),
        "y":  monthly["revenue"].values,
    })

    model = Prophet(
        yearly_seasonality      = False,
        weekly_seasonality      = False,
        daily_seasonality       = False,
        interval_width          = 0.80,
        changepoint_prior_scale = 0.05,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    # Predict 6 months ahead using monthly frequency
    future   = model.make_future_dataframe(periods=6, freq="MS")
    forecast = model.predict(future)

    hist_preds = forecast[forecast["ds"].isin(df_prophet["ds"])]["yhat"].values
    mae        = float(np.mean(np.abs(hist_preds - df_prophet["y"].values)))

    future_fc = forecast[forecast["ds"] > df_prophet["ds"].max()].copy()
    future_fc["yhat"]       = future_fc["yhat"].clip(lower=0)
    future_fc["yhat_lower"] = future_fc["yhat_lower"].clip(lower=0)
    future_fc["yhat_upper"] = future_fc["yhat_upper"].clip(lower=0)

    points = [
        {
            "date":      row["ds"].strftime("%Y-%m"),   # ← YYYY-MM not YYYY-MM-DD
            "predicted": round(float(row["yhat"]), 2),
            "lower":     round(float(row["yhat_lower"]), 2),
            "upper":     round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_fc.iterrows()
    ]

    return points, mae, "Prophet"


# ─────────────────────────────────────────────────────────────────────────────
#  Linear extrapolation fallback
# ─────────────────────────────────────────────────────────────────────────────

def _run_linear_fallback(monthly: pd.DataFrame) -> tuple[list, float, str]:
    values = monthly["revenue"].values.astype(float)
    x      = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend  = np.poly1d(coeffs)
    mae    = float(np.mean(np.abs(trend(x) - values)))

    last_date     = pd.Period(monthly["month"].iloc[-1], "M")
    monthly_preds = [max(0, float(trend(len(values) - 1 + i))) for i in range(1, 7)]
    points        = _build_monthly_forecast(monthly_preds, last_date)

    return points, mae, "Linear Trend"


# ─────────────────────────────────────────────────────────────────────────────
#  Monthly forecast output builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly_forecast(monthly_preds: list, last_history_month: pd.Period) -> list:
    """
    Converts list of predicted monthly revenue values into the forecast output
    format. Each point represents one full month — no daily interpolation.

    Output format:
      { "date": "2024-02", "predicted": 85000, "lower": 74800, "upper": 95200 }
    """
    points        = []
    current_month = last_history_month + 1

    for i, val in enumerate(monthly_preds):
        val          = max(0.0, float(val))
        month_period = current_month + i
        points.append({
            "date":      str(month_period),              # e.g. "2024-02"
            "predicted": round(val, 2),
            "lower":     round(val * (1 - CI_WIDTH), 2),
            "upper":     round(val * (1 + CI_WIDTH), 2),
        })

    return points