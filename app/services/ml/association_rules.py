import pandas as pd
from dataclasses import dataclass
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

@dataclass
class AssociationResult:
    rules:       list
    total_found: int
    mining_mode: str   # "product" | "category" | "empty"


def run_association_rules(
        df_basket:      pd.DataFrame,
        min_support:    float = 0.02,
        min_confidence: float = 0.3,
        top_n:          int   = 20,
) -> AssociationResult:
    """
    Mines association rules from basket data.

    Strategy (auto-selected):
    1. Product-level  — used when baskets are rich enough (avg > 2 items)
    2. Category-level — used for sparse datasets like Amazon reviews
                        where most users only bought 1-2 products.
                        Groups products by category per user and mines
                        category co-occurrence patterns instead.

    Args:
        df_basket:      basket table from preprocessing (must have 'items' column)
        min_support:    minimum frequency threshold (0.02 = appears in 2%+ of baskets)
        min_confidence: minimum confidence for a rule (0.3 = 30%+ co-occurrence)
        top_n:          max rules to return, sorted by lift
    """
    if df_basket.empty or "items" not in df_basket.columns:
        return AssociationResult(rules=[], total_found=0, mining_mode="empty")

    transactions = df_basket["items"].tolist()
    transactions = [t for t in transactions if isinstance(t, list) and len(t) > 1]

    # ── Decide mining mode ────────────────────────────────────────────────────
    # If we have enough multi-item baskets → product-level
    # Otherwise → category-level (works for Amazon-style data)
    avg_basket_size = (
        sum(len(t) for t in transactions) / len(transactions)
        if transactions else 0
    )

    if len(transactions) >= 50 and avg_basket_size >= 2:
        return _mine_rules(transactions, min_support, min_confidence, top_n, mode="product")

    # ── Category-level fallback ───────────────────────────────────────────────
    # df_basket items are product names/ids — we need the original df to get categories.
    # Check if category info was embedded (items may already be categories)
    # We detect this by checking if items look like category strings vs product IDs
    sample_items = transactions[0] if transactions else []
    looks_like_categories = (
        len(sample_items) > 0 and
        any(kw in " ".join(str(i) for i in sample_items).lower()
            for kw in ["electronic", "computer", "home", "kitchen", "cloth",
                       "book", "sport", "toy", "office", "music", "health"])
    )

    if looks_like_categories:
        return _mine_rules(transactions, min_support, min_confidence, top_n, mode="category")

    # ── Build category baskets from the items column ──────────────────────────
    # items are product names — extract category from the product name prefix
    # (best effort — works when product names include category hints)
    category_transactions = _extract_category_baskets(df_basket)

    if len(category_transactions) >= 10:
        return _mine_rules(
            category_transactions, min_support, min_confidence, top_n, mode="category"
        )

    # ── Last resort: try product-level with relaxed thresholds ────────────────
    if len(transactions) >= 10:
        return _mine_rules(
            transactions,
            min_support    = min_support / 4,
            min_confidence = min_confidence / 2,
            top_n          = top_n,
            mode           = "product",
        )

    return AssociationResult(rules=[], total_found=0, mining_mode="empty")


# ── Core mining logic ─────────────────────────────────────────────────────────

def _mine_rules(
        transactions:   list,
        min_support:    float,
        min_confidence: float,
        top_n:          int,
        mode:           str,
) -> AssociationResult:
    """
    Runs FP-Growth + association rules on a list of transactions.
    Automatically relaxes thresholds if nothing is found at first pass.
    """
    # Deduplicate items within each basket (categories especially can repeat)
    transactions = [list(dict.fromkeys(t)) for t in transactions]
    transactions = [t for t in transactions if len(t) > 1]

    if len(transactions) < 10:
        return AssociationResult(rules=[], total_found=0, mining_mode=mode)

    te       = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_enc   = pd.DataFrame(te_array, columns=te.columns_)

    # Try with original threshold, then relax if needed
    for support_multiplier in [1.0, 0.5, 0.25]:
        effective_support = min_support * support_multiplier
        frequent_itemsets = fpgrowth(
            df_enc,
            min_support  = effective_support,
            use_colnames = True,
        )
        if not frequent_itemsets.empty:
            break
    else:
        return AssociationResult(rules=[], total_found=0, mining_mode=mode)

    # Try with original confidence, then relax
    for conf_multiplier in [1.0, 0.5]:
        effective_conf = min_confidence * conf_multiplier
        try:
            rules_df = association_rules(
                frequent_itemsets,
                metric        = "confidence",
                min_threshold = effective_conf,
                num_itemsets  = len(frequent_itemsets),
            )
        except TypeError:
            # older mlxtend versions don't have num_itemsets
            rules_df = association_rules(
                frequent_itemsets,
                metric        = "confidence",
                min_threshold = effective_conf,
            )
        if not rules_df.empty:
            break
    else:
        return AssociationResult(rules=[], total_found=0, mining_mode=mode)

    rules_df    = rules_df.sort_values("lift", ascending=False)
    total_found = len(rules_df)
    rules_df    = rules_df.head(top_n)

    rules_list = [
        {
            "antecedents": sorted(list(row["antecedents"])),
            "consequents": sorted(list(row["consequents"])),
            "support":     round(float(row["support"]),    4),
            "confidence":  round(float(row["confidence"]), 4),
            "lift":        round(float(row["lift"]),       4),
            "mode":        mode,
        }
        for _, row in rules_df.iterrows()
    ]

    return AssociationResult(rules=rules_list, total_found=total_found, mining_mode=mode)


# ── Category basket builder ───────────────────────────────────────────────────

def _extract_category_baskets(df_basket: pd.DataFrame) -> list:
    """
    Builds category-level baskets from the items column.

    For Amazon data where items are product names, we use the first word
    or pipe-separated category prefix as a rough category signal.

    If the basket df has a 'category' column (passed through from preprocessing),
    we use that directly — much more accurate.
    """
    # Best case: category column was preserved in the basket df
    if "category" in df_basket.columns:
        cat_baskets = (
            df_basket.groupby(df_basket.index // 1)["category"]
            .apply(list).tolist()
        )
        return [list(set(b)) for b in cat_baskets if len(set(b)) > 1]

    # Fallback: try to infer category from item strings
    # This is a best-effort heuristic
    category_keywords = {
        "Electronics":          ["cable", "charger", "phone", "laptop", "tablet",
                                  "earphone", "headphone", "speaker", "camera"],
        "Computers":            ["keyboard", "mouse", "monitor", "ssd", "ram",
                                  "processor", "usb", "hub"],
        "HomeKitchen":          ["kitchen", "home", "cooker", "mixer", "fan",
                                  "light", "bulb", "iron"],
        "Clothing":             ["shirt", "trouser", "dress", "shoe", "sandal",
                                  "watch", "bag", "wallet"],
        "Books":                ["book", "novel", "guide", "manual"],
        "Sports":               ["sport", "gym", "fitness", "yoga", "cycle"],
        "Beauty":               ["cream", "lotion", "shampoo", "soap", "face",
                                  "hair", "skin"],
        "Toys":                 ["toy", "game", "puzzle", "lego", "doll"],
        "OfficeProducts":       ["pen", "pencil", "notebook", "stapler", "tape"],
    }

    def _infer_category(item: str) -> str:
        item_lower = str(item).lower()
        for cat, keywords in category_keywords.items():
            if any(kw in item_lower for kw in keywords):
                return cat
        return "Other"

    category_baskets = []
    for items in df_basket["items"]:
        if not isinstance(items, list):
            continue
        cats = list(set(_infer_category(item) for item in items))
        cats = [c for c in cats if c != "Other"]
        if len(cats) > 1:
            category_baskets.append(cats)

    return category_baskets