import io
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process, fuzz

CANONICAL_COLUMNS = {
    "customer_id": [
        "customer_id", "cust_id", "customerid", "client_id",
        "user_id", "customer id",
    ],
    "transaction_id": [
        "transaction_id", "order_id", "invoice_id", "txn_id",
        "invoiceno", "invoice_no", "invoice no", "invoice_number",
    ],
    "product_name": [
        "product_name", "item_name", "product", "item",
        "description", "product_description",
    ],
    "product_id": [
        "product_id", "item_id", "sku", "product_code",
        "stockcode", "stock_code", "stock code",
    ],
    "category": [
        "category", "product_category", "dept",
        "department", "type",
    ],
    "quantity": [
        "quantity", "qty", "units", "count", "amount",
    ],
    "price": [
        "price", "unit_price", "sale_price", "cost",
        "value", "unitprice", "unit price",
    ],
    "date": [
        "date", "purchase_date", "order_date",
        "transaction_date", "created_at",
        "invoicedate", "invoice_date", "invoice date",
    ],
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
    df = _load_file(file_bytes, content_type) # read data in form of bytes
    column_map = _map_columns(df.columns.tolist()) # mapped columns by converting columns into list first
    _validate_required_columns(column_map) # validated them
    reverse_map = {user_col : canonical for canonical, user_col in column_map.items()}
    df.rename(columns = reverse_map, inplace = True)
# Rename columns to canonical names
    reverse_map = {user_col: canonical for canonical, user_col in column_map.items()}
    df.rename(columns=reverse_map, inplace=True)

# TEMPORARY DEBUG — remove after fixing
    print(f"Column map: {column_map}")
    print(f"Columns after rename: {df.columns.tolist()}")
    print(f"Rows before cleaning: {len(df)}")   
    # Fix data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce") # errors="coerce" means it will mention NaN values where pandas cantparse the values with confidence
    df["quantity"] = pd.to_numeric(df["quantity"], errors = "coerce")
    df["price"] = pd.to_numeric(df["price"], errors = "coerce")
    df["customer_id"] = df["customer_id"].astype(str).str.strip() # pandas series fxn to access string fxn have to use str.fxn_name() in the continuity command
    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].astype(str).str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()
    # missing values handling, outlier removal
    # ── 4. Remove bad rows ─────────────────────────────────────────────────
    initial_rows = len(df)

# Drop rows missing critical values
    df.dropna(subset=["customer_id", "date", "price", "quantity"], inplace=True)

# Drop returns and zero-value rows
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]

# Remove price outliers using IQR method
    Q1  = df["price"].quantile(0.25)
    Q3  = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[df["price"].between(Q1 - 3 * IQR, Q3 + 3 * IQR)]

    rows_removed = initial_rows - len(df)

# ── ADD THIS CHECK ─────────────────────────────────────────────────────
    if len(df) < 10:
        raise ValueError(
                f"After cleaning, only {len(df)} valid rows remained. "
            f"The file had {rows_removed} rows removed due to missing customer IDs, "
            f"negative quantities, or extreme outliers. "
            f"Please check your data quality."
        )
    # Feature Engineering
    df["revenue"] = df["quantity"] * df["price"]
    df["day_of_week"] = df["date"].dt.dayofweek # .dt -> datetime accessor
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
def _load_file(file_bytes: bytes, content_type: str) -> pd.DataFrame:
    try:
        if "csv" in content_type:
            # Try UTF-8 first, fall back to latin-1 for special characters
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")
    return df
    
def _map_columns(user_columns: list) -> dict:
    """
    Maps user column names to canonical names using fuzzy matching.
    Uses normalized column names for matching but tracks originals.
    """
    all_aliases = {
        alias: canonical
        for canonical, aliases in CANONICAL_COLUMNS.items()
        for alias in aliases
    }

    column_map = {}
    used_canonicals = set()  # prevent two columns mapping to same canonical

    for original_col in user_columns:
        norm_col = original_col.lower().strip().replace(" ", "_")

        # Skip if this canonical is already mapped
        # Try exact match first
        if norm_col in all_aliases:
            canonical = all_aliases[norm_col]
            if canonical not in used_canonicals:
                column_map[canonical] = original_col
                used_canonicals.add(canonical)
            continue

        # Fuzzy match — but only accept if score is high enough
        match, score, _ = process.extractOne(
            norm_col,
            all_aliases.keys(),
            scorer=fuzz.token_sort_ratio
        )

        if score >= FUZZY_THRESHOLD:
            canonical = all_aliases[match]
            # Only map if this canonical hasn't been claimed yet
            if canonical not in used_canonicals:
                column_map[canonical] = original_col
                used_canonicals.add(canonical)

    return column_map

def _validate_required_columns(column_map : dict):
    """
    Raises ValueError if any required column wasn't found.
    """
    missing = REQUIRED_COLUMNS - set(column_map.keys()) # checks if required columns are there 
    if missing:
        raise ValueError(
            f"Could not find required columns : {missing}."
            f"Please make sure your file has columns for : "
            f"customer ID, quantity, price, and date."
        ) # if not raises error

def _build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the RFM feature table — one row per customer.

    Recency:   days since last purchase (lower = more recent = better)
    Frequency: number of unique transactions
    Monetary:  total spend
    customer_id | frequency | monetary
    ---------------------------------
    C1          |    5      |  1200
    C2          |    2      |   300
    """
    snapshot_date = df["date"].max() + pd.Timedelta(days=1) # recency >=1 not 0
    freq_col = "transaction_id" if "transaction_id" in df.columns else "date" # unique transaction because nunique will be applied after that

    # Calculate frequency and monetary first
    df_rfm = df.groupby("customer_id").agg(
        frequency = (freq_col,  "nunique"),
        monetary  = ("revenue", "sum"),
    ).reset_index()

    # Calculate recency separately and merge in
    # This avoids the pandas lambda deprecation warning
    last_purchase = (
        df.groupby("customer_id")["date"] # grouped by customer_id and only fetched date column
        .max() # max date
        .reset_index()
        .rename(columns={"date": "last_purchase_date"}) # renamed it 
    )

    df_rfm = df_rfm.merge(last_purchase, on="customer_id", how="left")
    df_rfm["recency"] = (snapshot_date - df_rfm["last_purchase_date"]).dt.days # calculated recency
    df_rfm.drop(columns=["last_purchase_date"], inplace=True)

    # Scale features for clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_rfm[["recency", "frequency", "monetary"]])
    df_rfm[["recency_scaled", "frequency_scaled", "monetary_scaled"]] = scaled # scaled the rfm features

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
    # gives grouped df wrt customer_id and transaction_id fot label_col attribute which is converted into list and the index are reset to previous one and label_col is renamed into items 
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