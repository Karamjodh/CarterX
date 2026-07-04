import pandas as pd
from dataclasses import dataclass
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


@dataclass
class AssociationResult:
    rules:       list
    total_found: int
    mining_mode: str   # "product" | "category" | "popular_fallback" | "empty"


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
    2. Category-level — used for sparse datasets (Amazon-style) where most
                        users only bought 1-2 products.
    3. Relaxed pass   — if both above fail, retries with halved thresholds.
    4. Popular-items  — final fallback when dataset is so sparse that NO
                        co-purchase rules can be mined at any threshold.
                        Returns top-N most purchased items ranked by
                        frequency so the frontend always has something
                        useful to show rather than a blank panel.
    """
    if df_basket.empty or "items" not in df_basket.columns:
        return AssociationResult(rules=[], total_found=0, mining_mode="empty")

    transactions = df_basket["items"].tolist()
    transactions = [t for t in transactions if isinstance(t, list) and len(t) > 1]

    # ── Decide mining mode ─────────────────────────────────────────────────
    avg_basket_size = (
        sum(len(t) for t in transactions) / len(transactions)
        if transactions else 0
    )

    if len(transactions) >= 50 and avg_basket_size >= 2:
        result = _mine_rules(transactions, min_support, min_confidence, top_n, mode="product")
        if result.total_found > 0:
            return result

    # ── Category-level fallback ────────────────────────────────────────────
    sample_items = transactions[0] if transactions else []
    looks_like_categories = (
        len(sample_items) > 0 and
        any(kw in " ".join(str(i) for i in sample_items).lower()
            for kw in ["electronic", "computer", "home", "kitchen", "cloth",
                       "book", "sport", "toy", "office", "music", "health"])
    )

    if looks_like_categories:
        result = _mine_rules(transactions, min_support, min_confidence, top_n, mode="category")
        if result.total_found > 0:
            return result

    # ── Build category baskets from items column ───────────────────────────
    category_transactions = _extract_category_baskets(df_basket)
    if len(category_transactions) >= 10:
        result = _mine_rules(
            category_transactions, min_support, min_confidence, top_n, mode="category"
        )
        if result.total_found > 0:
            return result

    # ── Last resort: relaxed product-level ────────────────────────────────
    if len(transactions) >= 10:
        result = _mine_rules(
            transactions,
            min_support    = min_support / 4,
            min_confidence = min_confidence / 2,
            top_n          = top_n,
            mode           = "product",
        )
        if result.total_found > 0:
            return result

    # ── Popular-items fallback ─────────────────────────────────────────────
    # Reaches here only when the dataset is so sparse (e.g. every customer
    # bought exactly 1 product) that no co-purchase signal exists at all.
    return _popular_items_fallback(df_basket, top_n)


# ── Core mining logic ──────────────────────────────────────────────────────────

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
    transactions = [list(dict.fromkeys(t)) for t in transactions]
    transactions = [t for t in transactions if len(t) > 1]

    if len(transactions) < 10:
        return AssociationResult(rules=[], total_found=0, mining_mode=mode)

    te       = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_enc   = pd.DataFrame(te_array, columns=te.columns_)

    # Try support thresholds: original → ÷2 → ÷4
    frequent_itemsets = None
    for support_multiplier in [1.0, 0.5, 0.25]:
        effective_support = min_support * support_multiplier
        fi = fpgrowth(df_enc, min_support=effective_support, use_colnames=True)
        if not fi.empty:
            frequent_itemsets = fi
            break

    if frequent_itemsets is None or frequent_itemsets.empty:
        return AssociationResult(rules=[], total_found=0, mining_mode=mode)

    # Try confidence thresholds: original → ÷2
    rules_df = None
    for conf_multiplier in [1.0, 0.5]:
        effective_conf = min_confidence * conf_multiplier
        try:
            df_rules = association_rules(
                frequent_itemsets,
                metric        = "confidence",
                min_threshold = effective_conf,
                num_itemsets  = len(frequent_itemsets),
            )
        except TypeError:
            df_rules = association_rules(
                frequent_itemsets,
                metric        = "confidence",
                min_threshold = effective_conf,
            )
        if not df_rules.empty:
            rules_df = df_rules
            break

    if rules_df is None or rules_df.empty:
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


# ── Popular-items fallback ─────────────────────────────────────────────────────

def _popular_items_fallback(df_basket: pd.DataFrame, top_n: int) -> AssociationResult:
    """
    When no co-purchase rules can be mined (fully sparse data — every customer
    bought only 1 product), fall back to ranking items by purchase frequency.

    Output format deliberately mirrors the association rules format so the
    frontend RulesTab can render it without any structural changes.
    Each "rule" is:  [] → [product]  with support = purchase_frequency
    and a special mode flag "popular_fallback" so the frontend can show
    a notice explaining why rules couldn't be mined.
    """
    all_items = []
    for items in df_basket["items"]:
        if isinstance(items, list):
            all_items.extend(items)
        # Also count single-item baskets that were filtered out earlier
    
    # Also scan single-item baskets (len == 1) which _mine_rules ignores
    single_items = df_basket["items"].tolist()
    for items in single_items:
        if isinstance(items, list):
            all_items.extend(items)

    if not all_items:
        return AssociationResult(rules=[], total_found=0, mining_mode="empty")

    item_counts  = pd.Series(all_items).value_counts()
    total_baskets = len(df_basket)

    rules_list = []
    for item, count in item_counts.head(top_n).items():
        support = round(count / total_baskets, 4) if total_baskets > 0 else 0.0
        rules_list.append({
            "antecedents": [],               # no antecedent — standalone recommendation
            "consequents": [str(item)],
            "support":     support,
            "confidence":  round(support, 4),
            "lift":        1.0,              # neutral lift — not a co-purchase signal
            "mode":        "popular_fallback",
            "purchase_count": int(count),
        })

    return AssociationResult(
        rules       = rules_list,
        total_found = len(rules_list),
        mining_mode = "popular_fallback",
    )


# ── Category basket builder ────────────────────────────────────────────────────

def _extract_category_baskets(df_basket: pd.DataFrame) -> list:
    """
    Builds category-level baskets from the items column.
    For Amazon data where items are product names, we use the first word
    as a rough category proxy.
    """
    category_transactions = []
    for items in df_basket["items"]:
        if not isinstance(items, list):
            continue
        categories = []
        for item in items:
            parts = str(item).strip().split()
            if parts:
                categories.append(parts[0].lower())
        categories = list(dict.fromkeys(categories))  # deduplicate, preserve order
        if len(categories) > 1:
            category_transactions.append(categories)
    return category_transactions