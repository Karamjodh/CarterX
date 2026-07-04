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

    X       = df_rfm[features].values
    n_rows  = len(X)

    # Need at least 10 customers to do anything meaningful
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

    # GMM component IDs are arbitrary — re-map them so cluster 0 is always
    # the highest-value group (sorted by mean monetary desc). This makes
    # the t-SNE cluster colours consistent across runs.
    component_monetary = {}
    for comp in np.unique(raw_components):
        mask = raw_components == comp
        component_monetary[comp] = df_rfm["monetary"].values[mask].mean()

    # sorted_comps[0] = component with highest avg monetary
    sorted_comps  = sorted(component_monetary, key=component_monetary.get, reverse=True)
    remap         = {old: new for new, old in enumerate(sorted_comps)}
    df_rfm["cluster"] = np.vectorize(remap.get)(raw_components)

    # ── Silhouette score (kept for backward compat with StatsTab / LLM prompt)
    if best_n >= 2:
        sil = silhouette_score(X, df_rfm["cluster"].values)
        sil = round(float(sil), 4)
    else:
        sil = 0.0

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
#  BIC sweep — core of the GMM approach
# ─────────────────────────────────────────────────────────────────────────────

def _bic_sweep(
    X:       np.ndarray,
    n_rows:  int,
) -> tuple:
    """
    Tries n_components = 2..max_k with multiple covariance types.
    Returns (best_gmm, best_n, list_of_bic_scores).

    Covariance types tried:
      "full"  — each component has its own full covariance matrix
                (most flexible, best for truly elliptical clusters)
      "tied"  — all components share one covariance matrix
                (good when clusters have similar shapes)
      "diag"  — diagonal covariance per component
                (faster, works well when features are independent)

    We try all three and keep whichever gives the lowest BIC overall.
    """
    # Max clusters: at least 2, at most 8, never more than n_rows // 5
    # (need at least 5 points per cluster for a meaningful GMM fit)
    max_k = max(2, min(8, n_rows // 5))

    COV_TYPES = ["full", "tied", "diag"]
    N_INIT    = 5      # multiple random initialisations per fit for stability
    MAX_ITER  = 300

    best_gmm   = None
    best_bic   = np.inf
    best_n     = 2
    bic_scores = []          # best BIC per n_components (for debug / frontend)

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
                logger.warning(
                    "GMM fit failed (n=%d, cov=%s): %s", n, cov_type, exc
                )

        if n_best_gmm is not None:
            bic_scores.append(round(n_best_bic, 2))
            if n_best_bic < best_bic:
                best_bic = n_best_bic
                best_gmm = n_best_gmm
                best_n   = n

    # Hard fallback — should never be needed but keeps pipeline alive
    if best_gmm is None:
        logger.error("All GMM fits failed — falling back to 2-component full GMM")
        best_gmm = GaussianMixture(
            n_components=2, covariance_type="full",
            random_state=42
        ).fit(X)
        best_n   = 2
        bic_scores = [best_gmm.bic(X)]

    return best_gmm, best_n, bic_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Profile builder — your existing label logic, kept intact
# ─────────────────────────────────────────────────────────────────────────────

def _build_profiles(df_rfm: pd.DataFrame, n_clusters: int) -> list:
    """
    Builds human-readable cluster profiles from raw RFM means.
    Label assignment logic is unchanged from the KMeans version —
    it already covers all the important segment archetypes.
    """
    profiles  = []
    overall_r = df_rfm["recency"].mean()
    overall_f = df_rfm["frequency"].mean()
    overall_m = df_rfm["monetary"].mean()
    max_r     = df_rfm["recency"].max()

    # Guard against max_r == 0 (all customers bought "today")
    max_r = max_r if max_r > 0 else 1

    for c in range(n_clusters):
        subset = df_rfm[df_rfm["cluster"] == c]

        if subset.empty:
            # GMM can sometimes produce an empty component — skip it
            logger.warning("Cluster %d is empty — skipping profile", c)
            continue

        r_mean = subset["recency"].mean()
        f_mean = subset["frequency"].mean()
        m_mean = subset["monetary"].mean()
        pct    = len(subset) / len(df_rfm) * 100

        # 0 = bought today (best), 1 = bought longest ago (worst)
        r_pct = r_mean / max_r

        # ── Label assignment ──────────────────────────────────────────────
        if r_pct < 0.2 and f_mean >= overall_f and m_mean >= overall_m:
            label = "Champions"
        elif r_pct < 0.3 and f_mean >= overall_f:
            label = "Loyal Customers"
        elif r_pct < 0.3 and f_mean < overall_f * 0.5:
            label = "New Customers"
        elif r_pct < 0.4 and m_mean >= overall_m * 1.2:
            label = "Potential Loyalists"
        elif r_pct > 0.7 and f_mean >= overall_f:
            label = "At Risk"
        elif r_pct > 0.7 and f_mean < overall_f:
            label = "Hibernating"
        elif r_pct > 0.85:
            label = "Lost Customers"
        elif m_mean >= overall_m * 1.5:
            label = "High Value"
        elif f_mean < overall_f * 0.4:
            label = "Low Engagement"
        else:
            label = f"Segment {c + 1}"

        profiles.append({
            "cluster_id":       c,
            "label":            label,
            "size":             int(len(subset)),
            "pct_of_customers": round(float(pct), 1),
            "avg_recency_days": round(float(r_mean), 1),
            "avg_frequency":    round(float(f_mean), 1),
            "avg_monetary":     round(float(m_mean), 2),
        })

    return profiles