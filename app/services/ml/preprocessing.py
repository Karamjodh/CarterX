import io
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# ── Column Aliases ─────────────────────────────────────────────────────────────
CANONICAL_COLUMNS = {
    "customer_id": [
        "customer_id", "cust_id", "customerid", "client_id",
        "user_id", "userid", "buyer_id",
        "customer_name", "customername", "buyer_name",      # covers "Customer Name"
    ],
    "transaction_id": [
        "transaction_id", "order_id", "invoice_id", "txn_id",
        "invoiceno", "invoice_no", "invoice_number",
        "review_id", "reviewid",
        "order_number", "order_no",                         # extra sales aliases
    ],
    "product_name": [
        "product_name", "item_name", "product", "item",
        "description", "product_description", "product_title", "title",
        "product_name", "item_description",
    ],
    "product_id": [
        "product_id", "item_id", "sku", "product_code",
        "stockcode", "stock_code", "asin",
    ],
    "category": [
        "category", "product_category", "dept", "department",
        "type", "product_type", "main_category",
    ],
    "quantity": [
        "quantity", "qty", "units", "count",
        "rating_count", "num_ratings",
        "quantity_sold", "units_sold", "quantity_ordered",  # extra sales aliases
    ],
    "price": [
        "price", "unit_price", "sale_price", "cost", "value",
        "unitprice", "selling_price", "discounted_price", "final_price",
        "total_sales", "total_revenue", "revenue",          # covers "Total Sales"
        "amount", "total_amount", "net_sales",
    ],
    "actual_price": [
        "actual_price", "original_price", "mrp", "list_price",
        "regular_price", "base_price", "market_price",
    ],
    "discount_pct": [
        "discount_percentage", "discount_pct", "discount",
        "discount_percent", "off_percentage",
    ],
    "rating": [
        "rating", "score", "stars", "review_score",
        "product_rating", "avg_rating", "user_rating",
    ],
    "date": [
        "date", "purchase_date", "order_date", "transaction_date",
        "created_at", "invoicedate", "invoice_date",
        "sale_date", "order_timestamp",
        "order_date", "delivery_date", "ship_date",         # extra sales aliases
    ],
    "user_name": [
        "user_name", "username", "reviewer_name", "reviewer",
    ],
    "review_title": [
        "review_title", "review_header", "review_summary",
    ],
    "review_content": [
        "review_content", "review_body", "review_text",
        "review", "comments", "feedback", "about_product",
    ],
    "img_link": [
        "img_link", "image_link", "image_url", "img_url",
    ],
    "product_link": [
        "product_link", "product_url", "item_url", "url", "link",
    ],
    "region": [
    "country", "region", "state", "city", "province",
    "location", "area", "territory", "zone", "market",
    "country_code", "geo", "geography", "place",
    "customer_location", "buyer_location",
    ],
}

# Strict — only these two are truly required
REQUIRED_COLUMNS = frozenset(["customer_id"])
FUZZY_THRESHOLD  = 80   # raised from 75 to avoid wrong matches


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class PreprocessingResult:
    df_clean:     pd.DataFrame
    df_rfm:       pd.DataFrame
    df_basket:    pd.DataFrame
    summary:      dict
    column_map:   dict
    trend_data:   dict
    dataset_type: str


# ── Public entry point ─────────────────────────────────────────────────────────
def run_preprocessing(file_bytes: bytes, content_type: str) -> PreprocessingResult:
    # 1. Load
    df = _load_file(file_bytes, content_type)

    # 2. Map columns
    column_map  = _map_columns(df.columns.tolist())
    logger.info(f"RAW COLUMNS: {df.columns.tolist()}")   # ← ADD THIS
    logger.info(f"MAPPED COLUMNS: {column_map}")           # ← ADD THIS
    _validate_required_columns(column_map)

    reverse_map = {user_col: canonical for canonical, user_col in column_map.items()}
    df.rename(columns=reverse_map, inplace=True)

    logger.info(f"Mapped columns: {column_map}")
    logger.info(f"Canonical columns present: {df.columns.tolist()}")

    # 3. Detect dataset type BEFORE any cleaning
    dataset_type = _detect_dataset_type(df)
    logger.info(f"Detected dataset type: {dataset_type}")

    # 4. For Amazon-style packed data: explode multi-value columns first
    if dataset_type == "review":
        df = _explode_review_dataset(df)
        logger.info(f"After exploding: {len(df)} rows")

    # 5. Clean all string formats (currency, commas, %, ratings) → numeric
    df = _clean_string_formats(df, dataset_type)

    # 6. Coerce types
    df = _coerce_types(df)

    # 7. Remove bad rows
    initial_rows = len(df)
    df = _clean_rows(df, dataset_type)
    rows_removed = initial_rows - len(df)
    logger.info(f"Rows after cleaning: {len(df)} (removed {rows_removed})")

    if len(df) < 10:
        raise ValueError(
            f"After cleaning, only {len(df)} valid rows remained "
            f"({rows_removed} rows removed). Please check your data quality."
        )

    # 8. Feature engineering
    df = _engineer_features(df, dataset_type)

    # 9. Build outputs
    df_rfm     = _build_rfm(df, dataset_type)
    df_basket  = _build_basket(df, dataset_type)
    summary    = _build_summary(df, dataset_type, rows_removed)
    trend_data = compute_trend_data(df, dataset_type)

    return PreprocessingResult(
        df_clean     = df,
        df_rfm       = df_rfm,
        df_basket    = df_basket,
        summary      = summary,
        column_map   = column_map,
        trend_data   = trend_data,
        dataset_type = dataset_type,
    )


# ── Dataset type detection ─────────────────────────────────────────────────────
def _detect_dataset_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "date" in cols and "quantity" in cols and "price" in cols:
        return "transactional"
    if "rating" in cols or "transaction_id" in cols:
        return "review"
    if "price" in cols and "product_id" in cols:
        return "catalog"
    return "transactional"


# ── Explode packed Amazon rows into one row per user ───────────────────────────
def _explode_review_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amazon dataset stores multiple users per row as comma-separated strings.
    This explodes them into one row per user-product interaction.

    Columns to explode: customer_id, transaction_id, user_name,
                        review_title, review_content
    Product-level columns (price, rating etc.) are broadcast to all rows.
    """
    # Columns that are packed (one value per user, comma-separated)
    packed_cols = []
    for col in ["customer_id", "transaction_id", "user_name",
                "review_title", "review_content"]:
        if col in df.columns:
            packed_cols.append(col)

    # Product-level columns (same value for all users of this product)
    product_cols = [c for c in df.columns if c not in packed_cols]

    rows = []
    for _, row in df.iterrows():
        # Split the first packed column to find how many users there are
        split_values = {
            col: [v.strip() for v in str(row[col]).split(",")]
            for col in packed_cols
            if row[col] is not None and str(row[col]).strip() not in ("", "nan", "None")
        }

        if not split_values:
            continue

        # Use customer_id length as the canonical count
        anchor = "customer_id" if "customer_id" in split_values else list(split_values.keys())[0]
        n = len(split_values[anchor])

        for i in range(n):
            new_row = {col: row[col] for col in product_cols}
            for col in packed_cols:
                vals = split_values.get(col, [])
                new_row[col] = vals[i] if i < len(vals) else None
            rows.append(new_row)

    df_exploded = pd.DataFrame(rows).reset_index(drop=True)
    return df_exploded


# ── Clean string formats before numeric coercion ───────────────────────────────
def _clean_string_formats(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Handles:
    - ₹399, ₹1,099  → 399.0, 1099.0
    - 64%            → 64.0
    - 24,269         → 24269.0
    - 4.2            → 4.2  (already fine)
    """
    # Price columns: strip ₹ $ € £ and commas
    for col in ["price", "actual_price"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[₹$€£,\s]", "", regex=True)
                .str.strip()
            )

    # Discount percentage: strip %
    if "discount_pct" in df.columns:
        df["discount_pct"] = (
            df["discount_pct"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

    # Rating count: strip commas  e.g. "24,269" → "24269"
    if "quantity" in df.columns:
        df["quantity"] = (
            df["quantity"].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    # Rating: handle "4.2 out of 5 stars" → "4.2"
    if "rating" in df.columns:
        df["rating"] = (
            df["rating"].astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
        )

    # Category: Amazon uses pipe-separated hierarchy — keep first level only
    if "category" in df.columns:
        df["category"] = (
            df["category"].astype(str)
            .str.split("|").str[0]
            .str.strip()
        )

    return df


# ── Type coercion ──────────────────────────────────────────────────────────────
def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = ["customer_id", "product_id", "product_name",
                "user_name", "review_title", "review_content",
                "transaction_id", "img_link", "product_link"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    num_cols = ["price", "actual_price", "discount_pct", "rating", "quantity"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ── Row cleaning ───────────────────────────────────────────────────────────────
def _clean_rows(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type == "transactional":
        df.dropna(subset=["customer_id", "date", "price", "quantity"], inplace=True)
        df = df[(df["quantity"] > 0) & (df["price"] > 0)]
        Q1, Q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
        IQR    = Q3 - Q1
        df     = df[df["price"].between(Q1 - 3 * IQR, Q3 + 3 * IQR)]

    elif dataset_type == "review":
        df.dropna(subset=["customer_id", "product_id"], inplace=True)
        # Remove "None" / "nan" strings that came from exploding
        df = df[~df["customer_id"].isin(["None", "nan", ""])]
        df = df[~df["product_id"].isin(["None", "nan", ""])]
        df.drop_duplicates(subset=["customer_id", "product_id"], keep="first", inplace=True)
        if "rating" in df.columns:
            df = df[df["rating"].between(1, 5) | df["rating"].isna()]

    elif dataset_type == "catalog":
        df.dropna(subset=["product_id", "price"], inplace=True)
        df = df[df["price"] > 0]

    return df.reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type == "transactional":
        df["revenue"]     = df["quantity"] * df["price"]
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"]       = df["date"].dt.month

    elif dataset_type == "review":
        # Revenue proxy: price × rating_count gives estimated sales volume
        if "price" in df.columns and "quantity" in df.columns:
            df["revenue"] = df["price"] * df["quantity"].fillna(1)
        elif "price" in df.columns:
            df["revenue"] = df["price"]
        else:
            df["revenue"] = 1.0

        if "actual_price" in df.columns and "price" in df.columns:
            df["discount_amount"] = (df["actual_price"] - df["price"]).clip(lower=0)

    elif dataset_type == "catalog":
        df["revenue"] = df["price"]

    return df


# ── RFM scaling ─────────────────────────────────────────────────────────────────
def _scale_rfm_features(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Properly scales RFM features for clustering.

    Steps:
    1. Log-transform monetary and frequency (both are right-skewed in
       almost all real-world retail/review data — a handful of customers
       spend or transact far more than everyone else).
    2. Use RobustScaler instead of StandardScaler — it scales using the
       median and IQR rather than the mean and standard deviation, so a
       few extreme outliers don't compress the rest of the distribution
       into a tiny band near zero.
    3. Recency is left un-logged — it's generally close to linear already
       and log-transforming it tends to hurt rather than help.

    Without this, StandardScaler alone tends to produce one giant
    "everyone normal" cluster plus tiny clusters made up of nothing but
    outliers — which is exactly the symptom of poorly separated clusters.
    """
    df_rfm = df_rfm.copy()

    # log1p = log(1 + x) — handles zero values safely (log(0) is undefined)
    monetary_log  = np.log1p(df_rfm["monetary"].clip(lower=0))
    frequency_log = np.log1p(df_rfm["frequency"].clip(lower=0))
    recency_raw   = df_rfm["recency"]

    scaler = RobustScaler()
    scaled = scaler.fit_transform(
        pd.DataFrame({
            "recency_raw":   recency_raw,
            "frequency_log": frequency_log,
            "monetary_log":  monetary_log,
        })
    )

    df_rfm["recency_scaled"]   = scaled[:, 0]
    df_rfm["frequency_scaled"] = scaled[:, 1]
    df_rfm["monetary_scaled"]  = scaled[:, 2]

    return df_rfm


# ── RFM building ───────────────────────────────────────────────────────────────
def _build_rfm(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type == "transactional":
        return _build_rfm_transactional(df)
    return _build_rfm_review(df)


def _build_rfm_transactional(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["date"].max() + pd.Timedelta(days=1)
    freq_col      = "transaction_id" if "transaction_id" in df.columns else "date"

    df_rfm = df.groupby("customer_id").agg(
        frequency=(freq_col,  "nunique"),
        monetary =("revenue", "sum"),
    ).reset_index()

    last_purchase = (
        df.groupby("customer_id")["date"].max()
        .reset_index().rename(columns={"date": "last_purchase_date"})
    )
    df_rfm = df_rfm.merge(last_purchase, on="customer_id", how="left")
    df_rfm["recency"] = (snapshot_date - df_rfm["last_purchase_date"]).dt.days
    df_rfm.drop(columns=["last_purchase_date"], inplace=True)

    # ── Log-transform + RobustScaler instead of plain StandardScaler ───────
    df_rfm = _scale_rfm_features(df_rfm)

    return df_rfm


def _build_rfm_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    R = inverted avg rating (low engagement = high recency number)
    F = number of unique products reviewed
    M = total estimated spend
    """
    agg_dict = {
        "frequency": ("product_id", "nunique"),
        "monetary":  ("revenue",    "sum"),
    }
    if "rating" in df.columns:
        agg_dict["avg_rating"] = ("rating", "mean")

    df_rfm = df.groupby("customer_id").agg(**agg_dict).reset_index()

    if "avg_rating" in df_rfm.columns:
        df_rfm["recency"] = ((5 - df_rfm["avg_rating"]) * 20).clip(lower=0).round()
    else:
        df_rfm["recency"] = 50

    df_rfm[["recency", "frequency", "monetary"]] = (
        df_rfm[["recency", "frequency", "monetary"]].fillna(0)
    )

    # ── Log-transform + RobustScaler instead of plain StandardScaler ───────
    df_rfm = _scale_rfm_features(df_rfm)

    return df_rfm


# ── Basket building ────────────────────────────────────────────────────────────
def _build_basket(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type == "transactional":
        label_col = "product_name" if "product_name" in df.columns else "product_id"
        if label_col not in df.columns:
            return pd.DataFrame({"items": []})
        group_col = "transaction_id" if "transaction_id" in df.columns else "date"
        df_basket = (
            df.groupby(["customer_id", group_col])[label_col]
            .apply(list).reset_index()
            .rename(columns={label_col: "items"})
        )
    else:
        # ✅ For review/catalog: use CATEGORY as the item, group by customer
        # This gives "customers who interacted with Electronics also used Computers"
        if "category" not in df.columns:
            return pd.DataFrame({"items": []})
        df_basket = (
            df.groupby("customer_id")["category"]
            .apply(lambda cats: list(set(cats)))  # unique categories per user
            .reset_index()
            .rename(columns={"category": "items"})
        )

    df_basket = df_basket[df_basket["items"].apply(len) > 1].reset_index(drop=True)
    return df_basket


# ── Summary building ───────────────────────────────────────────────────────────
def _build_summary(df: pd.DataFrame, dataset_type: str, rows_removed: int) -> dict:
    summary = {
        "total_customers": int(df["customer_id"].nunique()) if "customer_id" in df.columns else 0,
        "total_products":  int(df["product_id"].nunique())  if "product_id"  in df.columns else 0,
        "rows_removed":    rows_removed,
        "dataset_type":    dataset_type,
    }

    if dataset_type == "transactional":
        freq_col = "transaction_id" if "transaction_id" in df.columns else "date"
        summary["total_transactions"] = int(df[freq_col].nunique())
        summary["total_revenue"]      = round(float(df["revenue"].sum()), 2)
        summary["avg_order_value"]    = round(float(df["revenue"].mean()), 2)
        summary["date_start"]         = df["date"].min().strftime("%Y-%m-%d")
        summary["date_end"]           = df["date"].max().strftime("%Y-%m-%d")

    elif dataset_type == "review":
        summary["total_reviews"]   = len(df)
        summary["total_revenue"]   = round(float(df["revenue"].sum()), 2)
        summary["avg_order_value"] = round(float(df["revenue"].mean()), 2)
        if "rating" in df.columns:
            summary["avg_rating"]   = round(float(df["rating"].mean()), 2)
            summary["rating_range"] = f"{df['rating'].min()} – {df['rating'].max()}"
        if "discount_pct" in df.columns:
            summary["avg_discount_pct"] = round(float(df["discount_pct"].mean()), 1)

    if "category" in df.columns:
        if "revenue" in df.columns:
            top_cats = (
                df.groupby("category")["revenue"]
                .sum().sort_values(ascending=False)
                .head(5).round(2).to_dict()
            )
        else:
            top_cats = df["category"].value_counts().head(5).to_dict()
        summary["top_categories"] = top_cats

    return summary


# ── Trend data ─────────────────────────────────────────────────────────────────
def compute_trend_data(df: pd.DataFrame, dataset_type: str = "transactional") -> dict:
    # Check for any date column, not just transactional
    date_col = None
    for col in ["date", "rating_date", "review_date", "order_date", "purchase_date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        return _compute_static_trends(df)

    df = df.copy()
    df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)

    # Use discounted_price or actual_price if revenue column doesn't exist
    revenue_col = None
    for col in ["revenue", "discounted_price", "actual_price", "price"]:
        if col in df.columns:
            revenue_col = col
            break

    if revenue_col is None:
        return _compute_static_trends(df)

    monthly = (
        df.groupby("month")[revenue_col].sum().round(2).reset_index()
        .rename(columns={revenue_col: "total_revenue"})
        .sort_values("month")
    )

    category_monthly = {}
    if "category" in df.columns:
        cat_monthly = (
            df.groupby(["category", "month"])[revenue_col]
            .sum().round(2).reset_index().sort_values("month")
        )
        for cat in cat_monthly["category"].unique():
            cat_data = cat_monthly[cat_monthly["category"] == cat]
            category_monthly[cat] = cat_data[["month", revenue_col]].rename(
                columns={revenue_col: "revenue"}
            ).to_dict(orient="records")

    top_products = []
    product_col = next((c for c in ["product_name", "product_title", "name"] if c in df.columns), None)
    if product_col:
        top_products = (
            df.groupby(product_col)
            .agg(total_revenue=(revenue_col, "sum"))
            .round(2).sort_values("total_revenue", ascending=False)
            .head(10).reset_index()
            .rename(columns={product_col: "product_name"})
            .to_dict(orient="records")
        )

    monthly_vals = monthly["total_revenue"].tolist()
    mom_growth = 0
    if len(monthly_vals) >= 2:
        last, prev = monthly_vals[-1], monthly_vals[-2]
        mom_growth = round((last - prev) / prev * 100, 1) if prev > 0 else 0

    return {
        "monthly_revenue":  monthly.to_dict(orient="records"),
        "category_monthly": category_monthly,
        "top_products":     top_products,
        "mom_growth_pct":   mom_growth,
    }

def _compute_static_trends(df: pd.DataFrame) -> dict:
    label_col    = "product_name" if "product_name" in df.columns else "product_id"
    top_products = []

    if label_col in df.columns and "revenue" in df.columns:
        agg = {"total_revenue": ("revenue", "sum")}
        if "rating" in df.columns:
            agg["avg_rating"] = ("rating", "mean")
        top_products = (
            df.groupby(label_col).agg(**agg).round(2)
            .sort_values("total_revenue", ascending=False)
            .head(10).reset_index()
            .rename(columns={label_col: "product_name"})
            .to_dict(orient="records")
        )

    category_revenue = []
    if "category" in df.columns and "revenue" in df.columns:
        cat_rev = (
            df.groupby("category")["revenue"].sum().round(2)
            .sort_values(ascending=False).head(8).to_dict()
        )
        category_revenue = [{"category": k, "revenue": v} for k, v in cat_rev.items()]

    return {
        "monthly_revenue":  [],
        "category_monthly": {},
        "top_products":     top_products,
        "mom_growth_pct":   0,
        "category_revenue": category_revenue,
        "has_date_data":    False,
    }


# ── File loading ───────────────────────────────────────────────────────────────
def _load_file(file_bytes: bytes, content_type: str) -> pd.DataFrame:
    try:
        if "csv" in content_type:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")
    return df


# ── Column mapping ─────────────────────────────────────────────────────────────
def _map_columns(user_columns: list) -> dict:
    all_aliases = {
        alias: canonical
        for canonical, aliases in CANONICAL_COLUMNS.items()
        for alias in aliases
    }

    column_map      = {}
    used_canonicals = set()

    for original_col in user_columns:
        norm_col = original_col.lower().strip().replace(" ", "_").replace("-", "_")

        # Exact match first
        if norm_col in all_aliases:
            canonical = all_aliases[norm_col]
            if canonical not in used_canonicals:
                column_map[canonical] = original_col
                used_canonicals.add(canonical)
            continue

        # Fuzzy match
        result = process.extractOne(
            norm_col, all_aliases.keys(), scorer=fuzz.token_sort_ratio
        )
        if result:
            match, score, _ = result
            if score >= FUZZY_THRESHOLD:
                canonical = all_aliases[match]
                if canonical not in used_canonicals:
                    column_map[canonical] = original_col
                    used_canonicals.add(canonical)

    return column_map


def _validate_required_columns(column_map: dict):
    found_columns = set(column_map.keys())

    customer_aliases = {
        "customer_id", "user_id", "userid", "buyer_id",
        "client_id", "cust_id", "customerid"
    }
    has_customer = bool(found_columns & customer_aliases)

    if not has_customer:
        raise ValueError(
            "Could not identify a customer/user ID column in your file. "
            "Please make sure your data has a column like: "
            "customer_id, user_id, buyer_id, or userid."
        )

    useful_optional = {"product_id", "date", "price", "quantity", "rating"}
    missing_optional = useful_optional - found_columns
    if missing_optional:
        logger.warning(
            "Optional columns not found: %s — some features may be limited.",
            missing_optional
        )