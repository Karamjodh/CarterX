"""
CarterX — Forecasting Engine (v3)
─────────────────────────────────────────────────────────────────────────────
Generates revenue forecasts using LSTM, SARIMA, Prophet, or a seasonal-linear
fallback — model tier is chosen automatically from how much daily history
is available, with automatic cascading fallback if a tier's dependency or
fit fails.

WHAT CHANGED FROM v1 AND WHY
  v1 aggregated everything to MONTHLY totals, trained on those ~3-13 points,
  then stretched 6 monthly predictions into a daily line with
  np.linspace() — i.e. straight ramps between month-end values. That's why
  the forecast always looked flat/constant: there was no daily signal in
  the model at all, just linear interpolation wearing a daily costume.

  v2 builds a full DAILY revenue series (reindexed across every calendar
  day, missing days filled with 0 — e.g. this dataset is closed most
  Saturdays) and trains directly on it. Every model below now predicts
  real daily points, so weekly patterns (e.g. weekend dips) actually show
  up in both the history and the forecast.

STRATEGY (model selection is now based on daily coverage, not months) — v3:
  1. Build a daily revenue series (calendar-complete, gaps filled with 0)
  2. Pick a model tier automatically off the span of history:
       < 12 months   → Prophet   (best at squeezing signal out of short series;
                                   its priors/seasonality decomposition are more
                                   stable than SARIMA or LSTM on thin data)
       12–24 months  → SARIMA    (enough cycles to fit a real seasonal ARIMA
                                   order without overfitting; still too short
                                   for LSTM to learn reliably)
       > 24 months   → LSTM      (enough sequence data for a neural net to
                                   actually learn yearly + weekly structure
                                   instead of memorizing noise)
  3. AUTOMATED FALLBACK CHAIN: if the tier's model errors out (missing
     package, convergence failure, degenerate series, etc.) it automatically
     drops to the next simpler tier — LSTM → SARIMA → Prophet → Linear —
     rather than failing the whole request. `model_used` always reflects
     what actually ran, not what was requested.
  4. Output: daily forecast points for 30/60/90/180 days ahead,
             predicted directly (no monthly interpolation), with
             residual-based upper/lower confidence bands

OUTPUT CONTRACT (unchanged — what ForecastTab.js reads):
  {
    "model_used":    "LSTM" | "SARIMA" | "Prophet" | "Linear Trend",
    "history":       [{"date": "2023-01", "revenue": 12000}, ...],   # monthly, for the overview chart
    "forecast":      [{"date": "2024-02-01", "predicted": 13200,
                       "lower": 11800, "upper": 14600}, ...],        # daily, real granularity
    "horizons":      [30, 60, 90, 180],
    "mae":           450.2,
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

MIN_CI_WIDTH = 0.05   # floor so bands never collapse to zero on very clean data


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
        date_col, revenue_col, df = _resolve_date_and_revenue(df_clean)

        if date_col is None or revenue_col is None or df.empty:
            return ForecastResult(
                success       = True,
                model_used    = "none",
                has_date_data = False,
                warning       = (
                    "Not enough date-based data to generate a forecast. "
                    "Upload transactional data with dates to enable this feature."
                ),
            )

        monthly = _build_monthly_series(df, date_col, revenue_col)
        daily   = _build_daily_series(df, date_col, revenue_col)

        if monthly is None or len(monthly) < 3 or daily is None or len(daily) < 7:
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

        n_days = len(daily)
        tier   = _select_model_tier(n_days)
        forecast, mae, model_name, fallback_note = _run_with_fallback(tier, daily)

        warning = fallback_note
        if n_days < 60 and not warning:
            warning = (
                f"Forecast is based on only {n_days} days of history — "
                "treat the confidence band loosely and prefer shorter horizons."
            )

        return ForecastResult(
            success       = True,
            model_used    = model_name,
            history       = history,
            forecast      = forecast,
            horizons      = [30, 60, 90, 180],
            mae           = round(float(mae), 2),
            has_date_data = True,
            warning       = warning,
        )

    except Exception as e:
        logger.exception("Forecasting failed")
        return ForecastResult(success=False, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Date / revenue column resolution (shared by monthly + daily builders)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_date_and_revenue(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], pd.DataFrame]:
    date_col = None
    for col in ["date", "invoice_date", "order_date", "purchase_date",
                "created_at", "invoicedate", "sale_date", "order_timestamp"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning("Forecasting: no date column found in df_clean")
        return None, None, df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        logger.warning("Forecasting: all date values were NaT after coercion")
        return None, None, df

    revenue_col = None
    for col in ["revenue", "total_sales", "total_revenue",
                "amount", "total_amount", "net_sales"]:
        if col in df.columns:
            revenue_col = col
            logger.info("Forecasting: using revenue column '%s'", col)
            break

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
        return date_col, None, df

    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    return date_col, revenue_col, df


# ─────────────────────────────────────────────────────────────────────────────
#  Automatic model-tier selection + cascading fallback chain
# ─────────────────────────────────────────────────────────────────────────────

TIER_ORDER = ["LSTM", "SARIMA", "Prophet", "Linear Trend"]

_MODEL_FN = {
    "LSTM":         "_run_lstm_daily",
    "SARIMA":       "_run_sarima_daily",
    "Prophet":      "_run_prophet_daily",
    "Linear Trend": "_run_linear_seasonal_fallback",
}


def _select_model_tier(n_days: int) -> str:
    """Automatically pick a model tier from how much daily history we have.
       < 12mo -> Prophet | 12-24mo -> SARIMA | > 24mo -> LSTM
       (thresholds are in days so we don't care whether the data is
       reported as "monthly" or "daily" upstream)."""
    if n_days > 730:
        return "LSTM"
    if n_days >= 365:
        return "SARIMA"
    return "Prophet"


def _run_with_fallback(tier: str, daily: pd.DataFrame) -> tuple[list, float, str, Optional[str]]:
    """Run the chosen tier; on any failure (missing package, convergence
       error, degenerate series...) automatically step down to the next
       simpler tier. Linear Trend never fails (no external deps), so the
       chain always terminates. Returns a warning note if it had to
       downgrade, so the caller/UI can be honest about what actually ran."""
    start_idx = TIER_ORDER.index(tier)
    last_error = None

    for candidate in TIER_ORDER[start_idx:]:
        fn = globals()[_MODEL_FN[candidate]]
        try:
            forecast, mae, model_name = fn(daily)
            note = None
            if candidate != tier:
                note = (
                    f"{tier} was selected automatically for this data volume, but "
                    f"fell back to {model_name} ({last_error})."
                )
            return forecast, mae, model_name, note
        except Exception as e:
            logger.warning("%s failed (%s) — stepping down to next tier", candidate, e)
            last_error = str(e) or type(e).__name__
            continue

    # Should be unreachable — Linear Trend has no external deps — but guard anyway.
    forecast, mae, model_name = _run_linear_seasonal_fallback(daily)
    return forecast, mae, model_name, f"All higher-tier models failed ({last_error}); used Linear Trend."


# ─────────────────────────────────────────────────────────────────────────────
#  Monthly series — used ONLY for the history/overview chart now
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly_series(df: pd.DataFrame, date_col: str, revenue_col: str) -> Optional[pd.DataFrame]:
    df = df[df[revenue_col] > 0]
    if df.empty:
        return None

    df = df.copy()
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month")[revenue_col]
        .sum()
        .reset_index()
        .rename(columns={revenue_col: "revenue"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly if len(monthly) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
#  Daily series — this is what actually trains the model now.
#  Reindexed across every calendar day in range so closed days
#  (e.g. weekends) show up as real zeros instead of gaps — that's
#  where weekly seasonality lives.
# ─────────────────────────────────────────────────────────────────────────────

def _build_daily_series(df: pd.DataFrame, date_col: str, revenue_col: str) -> Optional[pd.DataFrame]:
    # Keep returns/negatives in the daily aggregate (a day can legitimately
    # net negative if refunds outweigh sales) — only row-level junk is dropped.
    daily = (
        df.groupby(df[date_col].dt.normalize())[revenue_col]
        .sum()
        .sort_index()
    )

    if daily.empty:
        return None

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)

    out = pd.DataFrame({"date": daily.index, "revenue": daily.values})
    return out if len(out) >= 7 else None


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM forecaster — daily windows, recursive daily prediction
# ─────────────────────────────────────────────────────────────────────────────

def _run_lstm_daily(daily: pd.DataFrame) -> tuple[list, float, str]:
    import tensorflow as tf  # noqa: F401 — raises ImportError if unavailable, caught by fallback chain
    from tensorflow import keras

    values = daily["revenue"].values.astype(float)
    n      = len(values)

    v_min, v_max = values.min(), values.max()
    scale        = v_max - v_min if v_max > v_min else 1.0
    values_norm  = (values - v_min) / scale

    WINDOW = min(30, n - 1)   # ~1 month of lookback so weekly + monthly shape is visible

    X, y = [], []
    for i in range(len(values_norm) - WINDOW):
        X.append(values_norm[i : i + WINDOW])
        y.append(values_norm[i + WINDOW])

    X = np.array(X).reshape(-1, WINDOW, 1)
    y = np.array(y)

    split          = max(1, int(len(X) * 0.85))
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
        batch_size      = max(1, min(32, len(X_train))),
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

    last_window  = values_norm[-WINDOW:].tolist()
    preds_norm   = []
    HORIZON_DAYS = 180

    for _ in range(HORIZON_DAYS):
        x_in = np.array(last_window[-WINDOW:]).reshape(1, WINDOW, 1)
        pred = float(model.predict(x_in, verbose=0)[0][0])
        preds_norm.append(pred)
        last_window.append(pred)

    preds     = [p * scale + v_min for p in preds_norm]
    last_date = pd.Timestamp(daily["date"].iloc[-1])
    ci_width  = max(MIN_CI_WIDTH, mae / (np.mean(np.abs(values)) + 1e-9))

    forecast = _daily_points(preds, last_date, ci_width)
    return forecast, mae, "LSTM"


# ─────────────────────────────────────────────────────────────────────────────
#  Prophet forecaster — trained directly on daily data, real weekly seasonality
# ─────────────────────────────────────────────────────────────────────────────

def _run_prophet_daily(daily: pd.DataFrame) -> tuple[list, float, str]:
    from prophet import Prophet  # raises ImportError if unavailable, caught by fallback chain

    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(daily["date"]),
        "y":  daily["revenue"].values,
    })

    span_days = (df_prophet["ds"].max() - df_prophet["ds"].min()).days

    model = Prophet(
        yearly_seasonality      = span_days >= 545,   # need ~1.5yrs before trusting a yearly cycle
        weekly_seasonality      = True,                # <- the fix: this is where the "flat line" signal lives
        daily_seasonality       = False,
        interval_width          = 0.80,
        changepoint_prior_scale = 0.05,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    future   = model.make_future_dataframe(periods=180, freq="D")
    forecast = model.predict(future)

    hist_preds = forecast[forecast["ds"].isin(df_prophet["ds"])]["yhat"].values
    mae        = float(np.mean(np.abs(hist_preds - df_prophet["y"].values)))

    future_fc               = forecast[forecast["ds"] > df_prophet["ds"].max()].copy()
    future_fc["yhat"]       = future_fc["yhat"].clip(lower=0)
    future_fc["yhat_lower"] = future_fc["yhat_lower"].clip(lower=0)
    future_fc["yhat_upper"] = future_fc["yhat_upper"].clip(lower=0)

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
#  SARIMA forecaster — the 12-24 month tier. Seasonal ARIMA with a
#  weekly (period=7) seasonal component, order picked automatically by a
#  small AIC grid search rather than hardcoded — this is the "automate it"
#  piece for this tier: no manual (p,d,q) tuning needed per dataset.
# ─────────────────────────────────────────────────────────────────────────────

def _run_sarima_daily(daily: pd.DataFrame) -> tuple[list, float, str]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    values = daily["revenue"].values.astype(float)
    last_date = pd.Timestamp(daily["date"].iloc[-1])

    # Small, cheap grid over (p,d,q) with a fixed weekly seasonal order —
    # picks whichever combo minimizes AIC on this series. Kept intentionally
    # small so this stays fast even on ~2 years of daily data.
    candidate_orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (0, 1, 1)]
    seasonal_order    = (1, 1, 1, 7)   # weekly seasonality — same signal Prophet gets from weekly_seasonality=True

    best_aic, best_order, best_fit = np.inf, None, None
    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    values,
                    order                      = order,
                    seasonal_order             = seasonal_order,
                    enforce_stationarity       = False,
                    enforce_invertibility      = False,
                )
                fit = model.fit(disp=False)
            if fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, order, fit
        except Exception:
            continue

    if best_fit is None:
        raise RuntimeError("SARIMA: no candidate order converged")

    in_sample = best_fit.fittedvalues
    mae = float(np.mean(np.abs(values[1:] - in_sample[1:])))  # skip first point (SARIMA warm-up artifact)

    HORIZON_DAYS = 180
    result       = best_fit.get_forecast(steps=HORIZON_DAYS)
    mean_fc      = result.predicted_mean
    ci           = result.conf_int(alpha=0.20)  # ~80% band, matches Prophet's interval_width

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    points = []
    for i, d in enumerate(future_dates):
        pred  = max(0.0, float(mean_fc[i]))
        lower = max(0.0, float(ci[i][0]))
        upper = max(0.0, float(ci[i][1]))
        points.append({
            "date":      d.strftime("%Y-%m-%d"),
            "predicted": round(pred, 2),
            "lower":     round(lower, 2),
            "upper":     round(upper, 2),
        })

    logger.info("SARIMA: selected order=%s seasonal_order=%s (AIC=%.1f)", best_order, seasonal_order, best_aic)
    return points, mae, "SARIMA"


# ─────────────────────────────────────────────────────────────────────────────
#  Seasonal-linear fallback — trend + day-of-week offsets, no external deps.
#  This replaces the old flat linear-then-interpolate path.
# ─────────────────────────────────────────────────────────────────────────────

def _run_linear_seasonal_fallback(daily: pd.DataFrame) -> tuple[list, float, str]:
    values = daily["revenue"].values.astype(float)
    dates  = pd.to_datetime(daily["date"])
    x      = np.arange(len(values))

    coeffs = np.polyfit(x, values, 1)
    trend  = np.poly1d(coeffs)
    resid  = values - trend(x)

    # Average residual per day-of-week → the "shape" the flat line was missing
    dow_offset = (
        pd.Series(resid, index=dates.dt.dayofweek)
        .groupby(level=0)
        .mean()
        .reindex(range(7), fill_value=0.0)
    )

    mae = float(np.mean(np.abs(resid)))

    last_date    = dates.iloc[-1]
    HORIZON_DAYS = 180
    future_x     = np.arange(len(values), len(values) + HORIZON_DAYS)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    preds = [
        max(0.0, float(trend(fx) + dow_offset[fd.dayofweek]))
        for fx, fd in zip(future_x, future_dates)
    ]

    ci_width = max(MIN_CI_WIDTH, mae / (np.mean(np.abs(values)) + 1e-9))
    forecast = _daily_points(preds, last_date, ci_width)
    return forecast, mae, "Linear Trend"


# ─────────────────────────────────────────────────────────────────────────────
#  Shared: build daily point dicts (used by LSTM + linear fallback;
#  Prophet builds its own since it already forecasts daily natively)
# ─────────────────────────────────────────────────────────────────────────────

def _daily_points(preds: list[float], last_history_date: pd.Timestamp, ci_width: float) -> list:
    future_dates = pd.date_range(last_history_date + pd.Timedelta(days=1), periods=len(preds), freq="D")
    points = []
    for d, val in zip(future_dates, preds):
        val = max(0.0, float(val))
        points.append({
            "date":      d.strftime("%Y-%m-%d"),
            "predicted": round(val, 2),
            "lower":     round(val * (1 - ci_width), 2),
            "upper":     round(val * (1 + ci_width), 2),
        })
    return points