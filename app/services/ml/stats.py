"""
CarterX — Stats Computation
─────────────────────────────────────────────────────────────────────────────
Builds the complete summary dict that StatsTab.js reads from.

Called in pipeline.py AFTER segmentation so it can combine:
  - prep  (PreprocessingResult) → raw data stats
  - seg   (SegmentationResult)  → cluster count, silhouette score
  - assoc (AssociationResult)   → rule count, mining mode

WHY A SEPARATE MODULE:
  Previously _build_summary() inside preprocessing.py built the summary
  dict, but it ran before segmentation so it couldn't include cluster stats.
  It also had gaps — missing fields for certain dataset types meant StatsTab
  showed "undefined" for mom_growth_pct, date_start, rows_removed etc.

  This function is the single source of truth for everything StatsTab needs.
  It guarantees every field is always present with a sensible default,
  regardless of dataset type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from app.services.ml.preprocessing import PreprocessingResult
    from app.services.ml.segmentation import SegmentationResult
    from app.services.ml.association_rules import AssociationResult

logger = logging.getLogger(__name__)


def compute_stats(
    prep:  "PreprocessingResult",
    seg:   "SegmentationResult",
    assoc: "AssociationResult",
) -> dict:
    """
    Builds one complete summary dict guaranteed to have every field
    StatsTab.js reads, with sensible defaults when data is unavailable.

    Args:
        prep:  result of run_preprocessing()
        seg:   result of run_segmentation()
        assoc: result of run_association_rules()

    Returns:
        dict that replaces prep.summary when stored in Insight.summary
    """
    df          = prep.df_clean
    dataset_type = prep.dataset_type

    # ── Core customer / transaction counts ────────────────────────────────
    total_customers = int(df["customer_id"].nunique()) if "customer_id" in df.columns else 0

    if "transaction_id" in df.columns:
        total_transactions = int(df["transaction_id"].nunique())
    elif dataset_type == "transactional" and "date" in df.columns:
        total_transactions = int(df.groupby(["customer_id", "date"]).ngroups)
    else:
        total_transactions = int(len(df))

    # ── Revenue ───────────────────────────────────────────────────────────
    if "revenue" in df.columns:
        total_revenue   = round(float(df["revenue"].sum()), 2)
        avg_order_value = round(float(df["revenue"].mean()), 2)
    elif "price" in df.columns:
        total_revenue   = round(float(df["price"].sum()), 2)
        avg_order_value = round(float(df["price"].mean()), 2)
    else:
        total_revenue   = 0.0
        avg_order_value = 0.0

    # ── Date range ────────────────────────────────────────────────────────
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not date_series.empty:
            date_start = date_series.min().strftime("%Y-%m-%d")
            date_end   = date_series.max().strftime("%Y-%m-%d")
            months_of_data = (
                (date_series.max().year - date_series.min().year) * 12
                + date_series.max().month - date_series.min().month + 1
            )
        else:
            date_start     = "N/A"
            date_end       = "N/A"
            months_of_data = 0
    else:
        date_start     = "N/A"
        date_end       = "N/A"
        months_of_data = 0

    # ── Rows removed ──────────────────────────────────────────────────────
    rows_removed = int(prep.summary.get("rows_removed", 0))

    # ── Top categories ────────────────────────────────────────────────────
    top_categories = _compute_top_categories(df)

    # ── Month-over-month growth ───────────────────────────────────────────
    mom_growth_pct = _compute_mom_growth(prep.trend_data)

    # ── Segmentation stats ────────────────────────────────────────────────
    segments_found   = int(seg.n_clusters)
    silhouette_score = float(seg.silhouette_score)

    # Segment quality label for StatsTab display
    if silhouette_score >= 0.5:
        silhouette_label = "strong"
    elif silhouette_score >= 0.3:
        silhouette_label = "moderate"
    else:
        silhouette_label = "weak"

    # ── Association rules stats ───────────────────────────────────────────
    rules_found  = int(assoc.total_found)
    mining_mode  = assoc.mining_mode

    # ── Dataset-specific extras ───────────────────────────────────────────
    extras = _dataset_extras(df, dataset_type, prep.summary)

    # ── Assemble final dict ───────────────────────────────────────────────
    stats = {
        # Core counts
        "total_customers":    total_customers,
        "total_transactions": total_transactions,
        "total_revenue":      total_revenue,
        "avg_order_value":    avg_order_value,

        # Date
        "date_start":    date_start,
        "date_end":      date_end,
        "months_of_data": months_of_data,

        # Data quality
        "rows_removed":  rows_removed,
        "dataset_type":  dataset_type,

        # Categories
        "top_categories": top_categories,

        # Trend
        "mom_growth_pct": mom_growth_pct,

        # Segmentation
        "segments_found":    segments_found,
        "silhouette_score":  round(silhouette_score, 4),
        "silhouette_label":  silhouette_label,

        # Association rules
        "rules_found": rules_found,
        "mining_mode": mining_mode,
    }

    # Merge dataset-specific extras (avg_rating, avg_discount_pct etc.)
    stats.update(extras)

    logger.info(
        "Stats computed: customers=%d  revenue=%.2f  segments=%d  rules=%d  mode=%s",
        total_customers, total_revenue, segments_found, rules_found, mining_mode,
    )

    return stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_top_categories(df: pd.DataFrame) -> dict:
    """Top 5 categories by revenue (or count if no revenue column)."""
    if "category" not in df.columns:
        return {}

    if "revenue" in df.columns:
        top = (
            df.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )
    else:
        top = (
            df["category"]
            .value_counts()
            .head(5)
            .to_dict()
        )

    # Convert keys to str so JSON serialisation never fails
    return {str(k): float(v) for k, v in top.items()}


def _compute_mom_growth(trend_data: dict) -> float:
    """
    Extract month-over-month growth from trend_data.
    Returns 0.0 if not computable (dateless datasets).
    """
    if not trend_data:
        return 0.0

    # Already computed by compute_trend_data() in preprocessing
    if "mom_growth_pct" in trend_data:
        return float(trend_data["mom_growth_pct"])

    # Recompute from monthly_revenue if available
    monthly = trend_data.get("monthly_revenue", [])
    if len(monthly) >= 2:
        revenues = [m.get("total_revenue", 0) for m in monthly]
        last, prev = revenues[-1], revenues[-2]
        if prev > 0:
            return round((last - prev) / prev * 100, 1)

    return 0.0


def _dataset_extras(df: pd.DataFrame, dataset_type: str, raw_summary: dict) -> dict:
    """Dataset-specific fields that StatsTab may optionally display."""
    extras = {}

    if dataset_type == "review":
        if "rating" in df.columns:
            extras["avg_rating"]   = round(float(df["rating"].mean()), 2)
            extras["rating_range"] = f"{df['rating'].min():.1f} – {df['rating'].max():.1f}"
        if "discount_pct" in df.columns:
            extras["avg_discount_pct"] = round(float(df["discount_pct"].mean()), 1)
        extras["total_reviews"] = int(len(df))

    elif dataset_type == "transactional":
        if "product_id" in df.columns:
            extras["total_products"] = int(df["product_id"].nunique())
        if "quantity" in df.columns:
            extras["avg_quantity"] = round(float(df["quantity"].mean()), 2)

    elif dataset_type == "catalog":
        if "product_id" in df.columns:
            extras["total_products"] = int(df["product_id"].nunique())
        if "price" in df.columns:
            extras["avg_price"]  = round(float(df["price"].mean()), 2)
            extras["price_range"] = f"{df['price'].min():.2f} – {df['price'].max():.2f}"

    return extras