import io
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process, fuzz

CANONICAL_COLUMNS ={
    "customer_id" : ["customer_id", "cust_id", "customerid", "client_id", "user_id"],
    "transaction_id" : ["transaction_id", "order_id", "invoice_id", "txn_id"],
    "product_name" : ["product_name", "item_name", "product", "item", "description"],
    "category" : ["category", "product_category", "dept", "department", "type"],
    "quantity" : ["quantity", "qty", "units", "count", "amount"],
    "price" : ["price", "unit_price", "sale_price", "cost", "value"],
    "date" : ["date", "purchase_date", "order_date", "transaction_date", "created_at"],
}
REQUIRED_COLUMNS = {"customer_id", "quantity", "price", "date"}
FUZZY_THRESHOLD = 75

@dataclass
class PreprocessingResult:
    """
    Everything the preprocessing stage produces
    The next stages (segmentation, rules) read from this object
    """
    df_clean : pd.DataFrame # cleaned transaction data
    df_rfm : pd.DataFrame # rfm values -> recency frequency and monetary based values
    df_basket : pd.DataFrame # transaction formatted for association rules 
    summary : dict # stats for dashboard
    column_map : dict # used columns mapped to canonical columns

def run_preprocessing(file_bytes: bytes, content_type : str) -> PreprocessingResult:
    """
    Full preprocessing pipeline.
    Takes raw file bytes and return clean data ready for ML Models.
    """
    # Load file, map columns and rename them
    df = _load_file(file_bytes, content_type)
    column_map = _map_columns(df.columns.tolist())
    _validate_required_columns(column_map)
    reverse_map = {user_col : canonical for canonical, user_col in column_map.items()}
    df.rename(columns = reverse_map, inplace = True)
    # Fix data types
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format = True, errors = "coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors = "coerce")
    df["price"] = pd.to_numeric(df["price"], errors = "coerce")
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].astype(str).str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()
    # missing values handling, outlier removal
    initial_rows = len(df)
    df.dropna(subset = ["customer_id", "date", "price", "quantity"], inplace = True)
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df["price"].between(Q1 - 3 * IQR, Q3 + 3 * IQR)]
    rows_removed = initial_rows - len(df)
    # Feature Engineering
    df["revenue"] = df["quantity"] * df["price"]
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df_rfm = _build_rfm(df)
    df_basket = _build_basket(df)
    summary = _build_summary(df, rows_removed)
    return PreprocessingResult(
        df_clean = df,
        df_rfm = df_rfm,
        df_basket = df_basket,
        summary = summary,
        column_map = column_map,
    )
# Private Helpers
def _load_file(file_bytes : bytes, content_type : str)-> pd.DataFrame:
    """
    Load CSV or XLSX bytes into a DataFrame.
    """
    try:
        if "csv" in content_type:
            return pd.read_csv(io.BytesIO(file_bytes))
        else:
            return pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read file : {e}")
    
def _map_columns(user_columns : list)-> dict:
    """
    Maps user column names to canonical names using fuzzy matching.

    Example:
        user has "cust_id" → maps to "customer_id"
        user has "purchase date" → maps to "date"

    Returns dict of {canonical_name: user_column_name}
    """
    all_aliases = {
        alias : canonical for canonical, aliases in CANONICAL_COLUMNS.items() for alias in aliases
    }
    normalized = {
        col.lower().strip().replace(" ", "_"): col for col in user_columns
    }
    column_map = {}
    for norm_col, original_col in normalized.items():
        if norm_col in all_aliases:
            canonical = all_aliases[norm_col]
            column_map[canonical] = original_col
            continue
        match, score, _ = process.extractOne(
            norm_col, all_aliases.keys(), scorer = fuzz.token_sort_ratio
        )
        if score >= FUZZY_THRESHOLD:
            canonical = all_aliases[match]
            column_map[canonical] = original_col

    return column_map

def _validate_required_columns(column_map : dict):
    """
    Raises ValueError if any required column wasn't found.
    """
    missing = REQUIRED_COLUMNS - set(column_map.keys())
    if missing:
        raise ValueError(
            f"Could not find required columns : {missing}."
            f"Please make sure your file has columns for : "
            f"customer ID, quantity, price, and date."
        )

def _build_rfm(df : pd.DataFrame) -> pd.DataFrame:
    """
    Builds the RFM feature table — one row per customer.

    Recency:   days since last purchase (lower = more recent = better)
    Frequency: number of unique transactions
    Monetary:  total spend

    Also adds scaled versions (mean=0, std=1) for the clustering algorithm.
    Clustering algorithms are sensitive to scale — without scaling,
    a customer with $10,000 spend would dominate over recency/frequency.
    """
    snapshot_date = df["date"].max() + pd.Timedelta(days = 1)
    freq_col = "transaction_id" if "transaction_id" in df.columns else "date"
    df_rfm = df.groupby("customer_id").agg(
        recency = ("date", lambda x : (snapshot_date - x.max()).days),
        frequency = (freq_col, "nunique"),
        monetary = ("revenue", sum),
        ).reset_index()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_rfm[["recency","frequency","monetary"]])
    df_rfm[["recency_scaled", "frequency_scaled", "monetary_scaled"]] = scaled
    return df_rfm

def _build_basket(df : pd.DataFrame) -> pd.DataFrame:
    """
    Formats data for association rule mining.

    Each row = one transaction with a list of items purchased.
    Example: [["Laptop", "Mouse", "Keyboard"], ["Phone", "Case"], ...]

    Uses transaction_id if available, otherwise groups by customer + date.
    """
    label_col = "product_name" if "product_name" in df.columns else "category"
    group_col = "transaction_id" if "transaction_id" in df.columns else "date"
    if label_col not in df.columns:
        return pd.DataFrame({"items":[]})
    df_basket = (df.groupby(["customer_id", group_col])[label_col].apply(list).reset_index().rename(columns = {label_col : "items"}))
    return df_basket

def _build_summary(df : pd.DataFrame, rows_removed : int) -> dict:
    """
    Computes summary statistics for the dashboard and LLM Prompt.
    """
    freq_col = "transaction_id" if "transaction_id" in df.columns else "date"
    summary =  {
        "total_customers" : int(df["customer_id"].nunique()),
        "total_transactions" : int(df[freq_col].nunique()),
        "total_revenue" : round(float(df["revenue"].sum()),2),
        "avg_order_value" : round(float(df["revenue"].mean()),2),
        "date_start" : df["date"].min().strftime("%Y-%m-%d"),
        "date_end" : df["date"].max().strftime("%Y-%m-%d"),
        "rows_removed" : rows_removed,
    }
    if "category" in df.columns:
        top_cats = (df.groupby("category")["revenue"].sum().sort_values(ascending = False).head(5).round(2).to_dict())
        summary["top_categories"] = top_cats

    return summary