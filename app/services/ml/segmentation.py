"""
CarterX — Segmentation Engine (GMM)
─────────────────────────────────────────────────────────────────────────────
Replaces KMeans with Gaussian Mixture Models (GMM) + BIC sweep.

WHY GMM OVER K-MEANS:
  KMeans forces spherical equal-variance clusters. On real RFM data the
  distribution is almost never spherical — "Champions" is a tight dense
  group while "Hibernating" is a wide scattered cloud. KMeans collapses
  these into 2 dominant blobs because silhouette score rewards clean
  spherical separation (and you'll almost always get it at k=2).

  GMM fits elliptical Gaussians so it can model those different shapes.
  BIC (Bayesian Information Criterion) is used instead of silhouette to
  pick k — it penalises model complexity so it won't overfit to noise,
  but it will discover k=4 or k=5 when genuine structure exists.

COMPATIBILITY CONTRACT (nothing downstream changes):
  ─ SegmentationResult dataclass: identical interface
  ─ df_rfm_labelled: still has "cluster" (int) column
  ─ df_rfm_labelled: still has recency_scaled / frequency_scaled /
    monetary_scaled columns  →  tsne.py won't break
  ─ cluster_profiles: same list-of-dicts format, same keys
  ─ n_clusters, silhouette_score: same fields
"""

import logging
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data contract — identical to old KMeans version
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    df_rfm_labelled:  pd.DataFrame   # needed by tsne.py
    n_clusters:       int
    silhouette_score: float
    cluster_profiles: list


# ─────────────────────────────────────────────────────────────────────────────
#  Label pool — ordered best → worst
#  Regardless of how many clusters GMM finds (2–8), every cluster
#  always gets a meaningful business name from this list.
# ─────────────────────────────────────────────────────────────────────────────

SEGMENT_LABELS = [
    "Champions",           # rank 0 — best composite RFM score
    "Loyal Customers",     # rank 1
    "Potential Loyalists", # rank 2
    "New Customers",       # rank 3
    "Promising",           # rank 4
    "At Risk",             # rank 5
    "Hibernating",         # rank 6
    "Lost Customers",      # rank 7 — worst RFM score
]


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point — called by pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(df_rfm: pd.DataFrame) -> SegmentationResult:
    """
    Finds natural customer segments using GMM + BIC sweep.

    Takes the scaled RFM table from preprocessing and returns
    labelled clusters with human-readable profiles.

    Args:
        df_rfm: DataFrame with columns:
                recency, frequency, monetary,
                recency_scaled, frequency_scaled, monetary_scaled
                (produced by preprocessing._scale_rfm_features)

    Returns:
        SegmentationResult with df_rfm_labelled (includes "cluster" column),
        n_clusters, silhouette_score, cluster_profiles list.
    """
    features = ["recency_scaled", "frequency_scaled", "monetary_scaled"]

    if not all(f in df_rfm.columns for f in features):
        raise ValueError(
            f"RFM dataframe missing scaled features. "
            f"Expected: {features}. Got: {df_rfm.columns.tolist()}"
        )

    X      = df_rfm[features].values
    n_rows = len(X)

    if n_rows < 10:
        raise ValueError(
            f"Only {n_rows} customers after preprocessing. "
            "Need at least 10 to segment."
        )

    # ── BIC sweep ────────────────────────────────────────────────────────────
    best_gmm, best_n, bic_scores = _bic_sweep(X, n_rows)

    # ── Assign hard cluster labels ────────────────────────────────────────────
    df_rfm         = df_rfm.copy()
    raw_components = best_gmm.predict(X)

    # Re-map arbitrary GMM component IDs so cluster 0 = highest monetary.
    # Consistent ordering means t-SNE colour mapping is stable across runs.
    component_monetary = {
        comp: df_rfm["monetary"].values[raw_components == comp].mean()
        for comp in np.unique(raw_components)
    }
    sorted_comps       = sorted(component_monetary, key=component_monetary.get, reverse=True)
    remap              = {old: new for new, old in enumerate(sorted_comps)}
    df_rfm["cluster"]  = np.vectorize(remap.get)(raw_components)

    # ── Silhouette score ──────────────────────────────────────────────────────
    sil = round(float(silhouette_score(X, df_rfm["cluster"].values)), 4) if best_n >= 2 else 0.0

    # ── Build profiles ────────────────────────────────────────────────────────
    profiles = _build_profiles(df_rfm, best_n)

    logger.info(
        "GMM segmentation done: n_clusters=%d  silhouette=%.4f  bic_scores=%s",
        best_n, sil, [round(b, 1) for b in bic_scores],
    )

    return SegmentationResult(
        df_rfm_labelled  = df_rfm,
        n_clusters       = best_n,
        silhouette_score = sil,
        cluster_profiles = profiles,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BIC sweep
# ─────────────────────────────────────────────────────────────────────────────

def _bic_sweep(X: np.ndarray, n_rows: int) -> tuple:
    """
    Tries n_components = 2..max_k across three covariance types.
    Returns (best_gmm, best_n, list_of_bic_scores).

    Covariance types:
      "full" — each component has its own full covariance matrix (most flexible)
      "tied" — all components share one covariance matrix
      "diag" — diagonal covariance per component (faster)
    """
    max_k     = max(2, min(8, n_rows // 5))
    COV_TYPES = ["full", "tied", "diag"]
    N_INIT    = 5
    MAX_ITER  = 300

    best_gmm   = None
    best_bic   = np.inf
    best_n     = 2
    bic_scores = []

    for n in range(2, max_k + 1):
        n_best_bic = np.inf
        n_best_gmm = None

        for cov_type in COV_TYPES:
            try:
                gmm = GaussianMixture(
                    n_components    = n,
                    covariance_type = cov_type,
                    n_init          = N_INIT,
                    max_iter        = MAX_ITER,
                    random_state    = 42,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < n_best_bic:
                    n_best_bic = bic
                    n_best_gmm = gmm
            except Exception as exc:
                logger.warning("GMM fit failed (n=%d, cov=%s): %s", n, cov_type, exc)

        if n_best_gmm is not None:
            bic_scores.append(round(n_best_bic, 2))
            if n_best_bic < best_bic:
                best_bic = n_best_bic
                best_gmm = n_best_gmm
                best_n   = n

    if best_gmm is None:
        logger.error("All GMM fits failed — falling back to 2-component full GMM")
        best_gmm   = GaussianMixture(n_components=2, covariance_type="full", random_state=42).fit(X)
        best_n     = 2
        bic_scores = [best_gmm.bic(X)]

    return best_gmm, best_n, bic_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Profile builder — rank-based dynamic labeling
# ─────────────────────────────────────────────────────────────────────────────

def _build_profiles(df_rfm: pd.DataFrame, n_clusters: int) -> list:
    """
    Builds human-readable cluster profiles with rank-based label assignment.

    WHY RANK-BASED INSTEAD OF THRESHOLD-BASED:
      Hard-coded thresholds like r_pct < 0.2 only fire for extreme clusters.
      When GMM finds 5-6 clusters (which it should on real data), most
      mid-range clusters fall through every condition → "Segment N".

      Rank-based assignment computes a composite RFM score per cluster,
      sorts best → worst, then maps labels in order from SEGMENT_LABELS.
      This guarantees every cluster always gets a real business name,
      regardless of how many clusters GMM discovers.

    COMPOSITE SCORE:
      score = monetary_norm + frequency_norm - recency_norm

      Each axis is normalised to [0,1] across clusters so no single
      dimension dominates. Recency is subtracted because lower recency
      (bought more recently) is better.
    """
    # ── Step 1: collect raw means per cluster ─────────────────────────────
    cluster_stats = []
    for c in range(n_clusters):
        subset = df_rfm[df_rfm["cluster"] == c]
        if subset.empty:
            logger.warning("Cluster %d is empty — skipping", c)
            continue
        cluster_stats.append({
            "cluster_id": c,
            "size":       len(subset),
            "r_mean":     subset["recency"].mean(),
            "f_mean":     subset["frequency"].mean(),
            "m_mean":     subset["monetary"].mean(),
            "pct":        len(subset) / len(df_rfm) * 100,
        })

    if not cluster_stats:
        return []

    # ── Step 2: normalise each RFM axis to [0, 1] across clusters ─────────
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]

    r_norm = _norm([s["r_mean"] for s in cluster_stats])
    f_norm = _norm([s["f_mean"] for s in cluster_stats])
    m_norm = _norm([s["m_mean"] for s in cluster_stats])

    # ── Step 3: composite score — higher = better customer ─────────────────
    for i, stat in enumerate(cluster_stats):
        # Recency subtracted: low recency (bought recently) = good
        stat["score"] = m_norm[i] + f_norm[i] - r_norm[i]

    # ── Step 4: sort best → worst, assign label by rank ────────────────────
    cluster_stats.sort(key=lambda s: s["score"], reverse=True)

    profiles = []
    for rank, stat in enumerate(cluster_stats):
        label = SEGMENT_LABELS[min(rank, len(SEGMENT_LABELS) - 1)]
        profiles.append({
            "cluster_id":       stat["cluster_id"],
            "label":            label,
            "size":             int(stat["size"]),
            "pct_of_customers": round(float(stat["pct"]), 1),
            "avg_recency_days": round(float(stat["r_mean"]), 1),
            "avg_frequency":    round(float(stat["f_mean"]), 1),
            "avg_monetary":     round(float(stat["m_mean"]), 2),
        })

    # Sort by cluster_id so index matches cluster int (frontend colour mapping)
    profiles.sort(key=lambda p: p["cluster_id"])
    return profiles