"""
t-SNE dimensionality reduction module.

t-SNE (t-distributed Stochastic Neighbor Embedding) is better than UMAP
for visualizing tight clusters — it preserves local structure more faithfully.
We use it as an alternative 2D visualization alongside UMAP.

Key differences vs UMAP:
- t-SNE is slower but often produces cleaner cluster separation visually
- t-SNE distances between clusters are NOT meaningful (only local structure)
- UMAP preserves more global structure
- We run both and let the frontend show either
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class TSNEResult:
    embedding_2d: list   # [{x, y, cluster_id, label, customer_id, monetary, recency, frequency}]
    explained_variance: float   # approximation — t-SNE doesn't have this natively


def run_tsne(
    df_rfm:   pd.DataFrame,
    profiles: list,
    perplexity: int = 30,
    n_iter:     int = 1000,
) -> TSNEResult:
    """
    Runs t-SNE on RFM features and returns 2D embedding for visualization.

    Args:
        df_rfm:     RFM dataframe with cluster column (from segmentation)
        profiles:   cluster profiles (for label lookup)
        perplexity: t-SNE perplexity — roughly "how many neighbors to consider"
                    Lower (5-15) = fine local structure
                    Higher (30-50) = broader structure
                    Rule of thumb: sqrt(n_samples)
        n_iter:     number of optimization iterations (more = better but slower)

    Returns:
        TSNEResult with 2D coordinates per customer
    """
    features = ["recency_scaled", "frequency_scaled", "monetary_scaled"]

    if not all(f in df_rfm.columns for f in features):
        raise ValueError("RFM dataframe missing scaled features. Run preprocessing first.")

    if "cluster" not in df_rfm.columns:
        raise ValueError("RFM dataframe missing cluster column. Run segmentation first.")

    X = df_rfm[features].values

    # Adjust perplexity if dataset is small
    # perplexity must be < n_samples
    effective_perplexity = min(perplexity, len(X) - 1, 50)

    logger.info(f"Running t-SNE on {len(X)} customers (perplexity={effective_perplexity})")

    tsne = TSNE(
        n_components  = 2,
        perplexity    = effective_perplexity,
        n_iter        = n_iter,
        random_state  = 42,
        learning_rate = "auto",
        init          = "pca",     # PCA init is more stable than random
        n_jobs        = -1,        # use all CPU cores
    )

    embedding = tsne.fit_transform(X)

    # Build label lookup from profiles
    label_map = {p["cluster_id"]: p["label"] for p in profiles}

    # Assemble result
    result_points = []
    for i, (x, y) in enumerate(embedding):
        row = df_rfm.iloc[i]
        result_points.append({
            "x":           round(float(x), 4),
            "y":           round(float(y), 4),
            "cluster_id":  int(row["cluster"]),
            "label":       label_map.get(int(row["cluster"]), f"Segment {int(row['cluster'])+1}"),
            "customer_id": str(row["customer_id"]),
            "monetary":    round(float(row["monetary"]) if pd.notna(row["monetary"]) else 0.0, 2),
            "recency":     int(row["recency"])   if pd.notna(row["recency"])   else 0,
            "frequency":   int(row["frequency"]) if pd.notna(row["frequency"]) else 0,
        })

    return TSNEResult(
        embedding_2d        = result_points,
        explained_variance  = 0.0,   # t-SNE doesn't produce explained variance
    )