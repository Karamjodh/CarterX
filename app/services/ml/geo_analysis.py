"""
CarterX — Geographic Analysis Service
──────────────────────────────────────────────────────────────────────────────
Analyzes customer and revenue distribution across geographic regions.

Works with any column that represents location after canonical mapping:
  "region" (canonical name) ← country, state, city, province, territory etc.

Also checks raw column names in case preprocessing didn't map them
(e.g. if the column was named something unusual).

No external APIs needed. Outputs clean dicts ready for JSON serialization.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoAnalysisResult:
    has_geo_data:         bool
    geo_column:           str    = ""
    region_stats:         list   = field(default_factory=list)
    region_growth:        list   = field(default_factory=list)
    top_regions:          list   = field(default_factory=list)
    regional_segments:    dict   = field(default_factory=dict)
    regional_products:    dict   = field(default_factory=dict)
    market_concentration: dict   = field(default_factory=dict)
    summary:              dict   = field(default_factory=dict)


def _empty_result(geo_col: str = "", message: str = "") -> GeoAnalysisResult:
    return GeoAnalysisResult(
        has_geo_data         = False,
        geo_column           = geo_col,
        region_stats         = [],
        region_growth        = [],
        top_regions          = [],
        regional_segments    = {},
        regional_products    = {},
        market_concentration = {},
        summary              = {"message": message or "No geographic data found."},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point — called by pipeline.py after segmentation
# ─────────────────────────────────────────────────────────────────────────────

def run_geo_analysis(
    df:               pd.DataFrame,
    cluster_profiles: list,
    dataset_type:     str = "transactional",
) -> GeoAnalysisResult:
    """
    Main entry point for geographic analysis.

    Args:
        df:               prep.df_clean (cleaned DataFrame from preprocessing)
        cluster_profiles: seg.cluster_profiles (for regional segment breakdown)
        dataset_type:     "transactional" | "review" | "catalog"

    Returns:
        GeoAnalysisResult with all geographic insights serializable to JSON.
    """
    # ── 1. Find geo column ─────────────────────────────────────────────────
    geo_col = _find_geo_column(df)

    if geo_col is None:
        logger.info("Geo analysis: no geographic column found")
        return _empty_result(message="No geographic data found in dataset.")

    logger.info("Geo analysis: using column '%s'", geo_col)

    # ── 2. Clean geo values ────────────────────────────────────────────────
    df = df.copy()
    df[geo_col] = (
        df[geo_col].astype(str)
        .str.strip()
        .str.title()
    )
    # Drop null/generic values
    invalid = {"Nan", "None", "Unknown", "N/A", "Na", "", "Unspecified", "Other", "Null"}
    df = df[~df[geo_col].isin(invalid)].reset_index(drop=True)

    if len(df) < 5:
        return _empty_result(geo_col, "Not enough valid geographic data after cleaning.")

    # ── 3. Run all analyses ────────────────────────────────────────────────
    region_stats         = _compute_region_stats(df, geo_col)
    region_growth        = _compute_region_growth(df, geo_col)
    top_regions          = _get_top_regions(region_stats)
    regional_segments    = _compute_regional_segments(df, geo_col, cluster_profiles)
    regional_products    = _compute_regional_products(df, geo_col)
    market_concentration = _compute_market_concentration(region_stats)
    summary              = _build_geo_summary(df, geo_col, region_stats, market_concentration)

    return GeoAnalysisResult(
        has_geo_data         = True,
        geo_column           = geo_col,
        region_stats         = region_stats,
        region_growth        = region_growth,
        top_regions          = top_regions,
        regional_segments    = regional_segments,
        regional_products    = regional_products,
        market_concentration = market_concentration,
        summary              = summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Column finder
# ─────────────────────────────────────────────────────────────────────────────

def _find_geo_column(df: pd.DataFrame) -> str | None:
    """
    Finds geographic column. Checks canonical 'region' name first
    (set by preprocessing CANONICAL_COLUMNS mapping), then falls
    back to checking common raw column names that might not have
    been mapped.
    """
    # Canonical name — preprocessing maps country/state/city → "region"
    if "region" in df.columns:
        if df["region"].nunique() >= 2:
            return "region"

    # Raw names — in case preprocessing didn't map them
    raw_geo_names = [
        "country", "state", "city", "province", "location",
        "area", "territory", "zone", "market", "geo",
        "geography", "place", "customer_location",
        "Country", "State", "City", "Location",
    ]
    for col in raw_geo_names:
        if col in df.columns:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 500:   # sanity check
                return col

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_region_stats(df: pd.DataFrame, geo_col: str) -> list:
    """Per-region revenue, customers, transactions, avg order value."""
    revenue_col = _find_revenue_col(df)

    agg: dict = {}
    if revenue_col:
        agg["total_revenue"] = (revenue_col, "sum")
        agg["avg_order_value"] = (revenue_col, "mean")

    if "customer_id" in df.columns:
        agg["unique_customers"] = ("customer_id", "nunique")

    agg["total_transactions"] = (geo_col, "count")

    if not agg:
        return []

    stats = (
        df.groupby(geo_col)
        .agg(**agg)
        .reset_index()
        .rename(columns={geo_col: "region"})
    )

    # Round numeric columns
    for col in ["total_revenue", "avg_order_value"]:
        if col in stats.columns:
            stats[col] = stats[col].round(2)

    # Sort by revenue desc (or transactions if no revenue)
    sort_col = "total_revenue" if "total_revenue" in stats.columns else "total_transactions"
    stats = stats.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Add revenue share % and cumulative share
    if "total_revenue" in stats.columns:
        total = stats["total_revenue"].sum()
        stats["revenue_share_pct"] = (
            (stats["total_revenue"] / total * 100).round(1) if total > 0 else 0
        )
        stats["cumulative_share_pct"] = stats["revenue_share_pct"].cumsum().round(1)

    return stats.to_dict("records")


def _compute_region_growth(df: pd.DataFrame, geo_col: str) -> list:
    """Monthly revenue trend per region — only when date data is present."""
    date_col    = _find_date_col(df)
    revenue_col = _find_revenue_col(df)

    if date_col is None or revenue_col is None:
        return []

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return []

    df["month"] = df[date_col].dt.to_period("M").astype(str)

    monthly = (
        df.groupby([geo_col, "month"])[revenue_col]
        .sum()
        .round(2)
        .reset_index()
        .rename(columns={geo_col: "region", revenue_col: "total_revenue"})
        .sort_values("month")
    )

    # Return as dict keyed by region for easy frontend lookup
    result = []
    for region, grp in monthly.groupby("region"):
        result.append({
            "region":  region,
            "monthly": grp[["month", "total_revenue"]].to_dict("records"),
        })

    return result


def _get_top_regions(region_stats: list, n: int = 10) -> list:
    """Top N regions by revenue."""
    return region_stats[:n]


def _compute_regional_segments(
    df:               pd.DataFrame,
    geo_col:          str,
    cluster_profiles: list,
) -> dict:
    """
    Segment distribution per region.
    Only meaningful when customer_id exists in both df and cluster_profiles.
    Returns dict: { region: { segment_label: count } }
    """
    if "customer_id" not in df.columns or not cluster_profiles:
        return {}

    # Build customer → region map
    cust_region = (
        df.groupby("customer_id")[geo_col]
        .first()
        .reset_index()
        .rename(columns={geo_col: "region"})
    )

    # We don't have customer→segment mapping in df_clean directly,
    # so we return customer counts per region as a proxy
    region_custs = (
        cust_region.groupby("region")["customer_id"]
        .count()
        .reset_index()
        .rename(columns={"customer_id": "customer_count"})
        .sort_values("customer_count", ascending=False)
        .head(20)
    )

    return region_custs.to_dict("records")


def _compute_regional_products(df: pd.DataFrame, geo_col: str) -> dict:
    """Top 5 products by revenue per region."""
    product_col = _find_product_col(df)
    revenue_col = _find_revenue_col(df)

    if product_col is None or revenue_col is None:
        return {}

    result = {}
    top_regions = df[geo_col].value_counts().head(10).index.tolist()

    for region in top_regions:
        region_df = df[df[geo_col] == region]
        top_prods = (
            region_df.groupby(product_col)[revenue_col]
            .sum()
            .round(2)
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .rename(columns={product_col: "product_name", revenue_col: "total_revenue"})
            .to_dict("records")
        )
        if top_prods:
            result[region] = top_prods

    return result


def _compute_market_concentration(region_stats: list) -> dict:
    """
    Herfindahl-Hirschman Index (HHI) — measures market concentration.
    HHI > 2500: highly concentrated
    HHI 1500-2500: moderately concentrated
    HHI < 1500: competitive
    """
    if not region_stats or "revenue_share_pct" not in region_stats[0]:
        return {}

    shares = [r["revenue_share_pct"] for r in region_stats]
    hhi    = round(sum(s ** 2 for s in shares), 1)

    if hhi > 2500:
        label = "Highly concentrated"
        desc  = "Revenue is dominated by a small number of regions."
    elif hhi > 1500:
        label = "Moderately concentrated"
        desc  = "A few regions drive most of the revenue."
    else:
        label = "Competitive"
        desc  = "Revenue is spread relatively evenly across regions."

    # Top 3 share
    top3_share = round(sum(shares[:3]), 1) if len(shares) >= 3 else round(sum(shares), 1)
    top1_share = round(shares[0], 1) if shares else 0

    return {
        "hhi":          hhi,
        "label":        label,
        "description":  desc,
        "top1_share":   top1_share,
        "top3_share":   top3_share,
        "n_regions":    len(region_stats),
    }


def _build_geo_summary(
    df:                  pd.DataFrame,
    geo_col:             str,
    region_stats:        list,
    market_concentration: dict,
) -> dict:
    """High-level geographic summary for KPI cards."""
    n_regions   = df[geo_col].nunique()
    top_region  = region_stats[0]["region"] if region_stats else "N/A"
    top_revenue = region_stats[0].get("total_revenue", 0) if region_stats else 0
    top_share   = region_stats[0].get("revenue_share_pct", 0) if region_stats else 0

    return {
        "total_regions":      n_regions,
        "top_region":         top_region,
        "top_region_revenue": top_revenue,
        "top_region_share":   top_share,
        "hhi":                market_concentration.get("hhi", 0),
        "concentration_label":market_concentration.get("label", ""),
        "geo_column_used":    geo_col,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Column finder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_revenue_col(df: pd.DataFrame) -> str | None:
    for col in ["revenue", "total_sales", "total_revenue", "amount", "price"]:
        if col in df.columns:
            return col
    return None


def _find_date_col(df: pd.DataFrame) -> str | None:
    for col in ["date", "invoice_date", "order_date", "purchase_date", "created_at"]:
        if col in df.columns:
            return col
    return None


def _find_product_col(df: pd.DataFrame) -> str | None:
    for col in ["product_name", "product_id", "item_name", "product", "item"]:
        if col in df.columns:
            return col
    return None