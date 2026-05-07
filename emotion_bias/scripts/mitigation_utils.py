"""
mitigation_utils.py - Shared utilities for the emotion-guided bias mitigation pipeline.

Implements the mathematical pieces described in the mitigation outline:
  - Emotion feature projection:  s = V^T h   (hidden-state -> emotion feature vector)
  - Protected subspace construction: PCA in feature space, then QR orthonormalization
      to get an orthonormal basis U (d, k) in hidden-state space.
  - Bias direction methods: diff / LDA / PLS / LogReg (all in feature space),
      lifted to hidden space via d_bias = V @ w.
  - Residual decomposition: d_bias = d_parallel + d_perp against a protected basis U.
  - Risk score for optional conditional steering: r = w^T (V^T h).

Notation
  d = hidden_dim, m = num_emotions, k = protected subspace dim.
  V   : (d, m)  columns are unit-norm emotion vectors at a given layer.
  s_i : (m,)    emotion features of example i at that layer.
  w   : (m,)    feature-space weights from a bias-direction method.
  U   : (d, k)  orthonormal basis for the protected subspace in hidden space.

All math is done in CPU float32 for numerical stability; downstream code is
responsible for moving tensors back to GPU at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Emotion-vector matrix utilities
# ============================================================

def build_V_matrix(
    vectors_at_layer: Dict[str, torch.Tensor],
    emotions_order: List[str],
) -> torch.Tensor:
    """
    Assemble emotion vectors into a (d, m) matrix V whose columns are the
    unit-norm emotion directions at a given layer.

    Args:
        vectors_at_layer: dict emotion_name -> unit vector of shape (d,)
        emotions_order:   canonical order of emotion names (same order must be
                          used everywhere downstream, e.g. when stacking X).
    Returns:
        V: tensor (d, m), float32
    """
    cols = [vectors_at_layer[e].float() for e in emotions_order]
    V = torch.stack(cols, dim=1)  # (d, m)
    return V


def hidden_to_features(
    hidden_states: torch.Tensor,  # (N, d)
    V: torch.Tensor,              # (d, m)
) -> torch.Tensor:
    """Project N hidden states onto m emotion directions: s = V^T h. Returns (N, m)."""
    return hidden_states.float() @ V.float()


def stack_features_from_projections(
    per_example_proj: List[Dict[str, float]],
    emotions_order: List[str],
) -> np.ndarray:
    """
    Convert a list of {emotion: proj} dicts (as stored in your probe results)
    into a (N, m) numpy array in the specified emotion order.
    Examples where any emotion is missing are skipped.
    """
    rows = []
    for proj in per_example_proj:
        if all(e in proj for e in emotions_order):
            rows.append([proj[e] for e in emotions_order])
    return np.array(rows, dtype=np.float32)


# ============================================================
# Protected subspace (Stage A)
# ============================================================

def fit_pca_on_features(
    X_features: np.ndarray,  # (N, m)
    standardize: bool = True,
) -> Tuple[PCA, Optional[StandardScaler]]:
    """Fit PCA on emotion features. Returns (pca, scaler); scaler is None if standardize=False."""
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(X_features)
        X_work = scaler.transform(X_features)
    else:
        X_work = X_features
    n_components = min(X_work.shape)
    pca = PCA(n_components=n_components).fit(X_work)
    return pca, scaler


def select_k_variance(pca: PCA, gamma: float = 0.95) -> int:
    """Smallest k with cumulative explained variance >= gamma."""
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, gamma) + 1)
    return max(1, min(k, len(cum)))


def select_k_task_validation(
    X_features: np.ndarray,       # (N, m)
    y_task: np.ndarray,           # (N,) integer labels for the non-bias emotion task
    pca: PCA,
    scaler: Optional[StandardScaler] = None,
    gamma: float = 0.95,
    cv: int = 3,
) -> Tuple[int, Dict[object, float]]:
    """
    Smallest k such that CV accuracy of a small classifier on the top-k PC
    projections is >= gamma * CV accuracy on the full feature space.

    Returns:
        chosen_k: int
        per_k_accuracy: dict mapping k -> mean CV accuracy, plus the sentinel
                        key "full" for the full-feature-space baseline.
    """
    X_work = scaler.transform(X_features) if scaler is not None else X_features

    clf_full = LogisticRegression(max_iter=500, random_state=42)
    perf_full = cross_val_score(clf_full, X_work, y_task, cv=cv, scoring="accuracy").mean()

    per_k: Dict[object, float] = {"full": float(perf_full)}
    n_components = pca.n_components_
    chosen_k = n_components  # fallback: use all
    for k in range(1, n_components + 1):
        Z = X_work @ pca.components_[:k].T  # (N, k)
        clf = LogisticRegression(max_iter=500, random_state=42)
        try:
            perf_k = cross_val_score(clf, Z, y_task, cv=cv, scoring="accuracy").mean()
        except ValueError:
            perf_k = 0.0
        per_k[k] = float(perf_k)
        if perf_k >= gamma * perf_full and chosen_k == n_components:
            chosen_k = k
            # keep looping to record per-k accuracies
    return chosen_k, per_k


def build_protected_basis_in_hidden_space(
    V: torch.Tensor,              # (d, m)
    pca_components: np.ndarray,   # (n_components, m) -- each row is a PC in feature space
    k: int,
) -> torch.Tensor:
    """
    Lift top-k PCs from feature space into hidden space and orthonormalize via QR.

    A feature-space PC (shape (m,)) corresponds to a hidden-space direction V @ pc.
    We stack the k lifted directions as columns of B in (d, k) and QR-decompose
    to obtain an orthonormal basis U for their span.

    Returns:
        U: (d, k) orthonormal
    """
    PC_top = pca_components[:k].T  # (m, k)
    B = V.float() @ torch.from_numpy(PC_top).float()  # (d, k)
    Q, _ = torch.linalg.qr(B, mode="reduced")
    return Q  # (d, k)


def projection_matrix_from_basis(U: torch.Tensor) -> torch.Tensor:
    """Return P_S = U U^T for orthonormal U. Dense (d, d); only use when d is small enough."""
    return U.float() @ U.float().T


# ============================================================
# Bias direction methods (Stage B) -- all operate in feature space
# ============================================================

def bias_weights_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method 1: mean(X | y=1) - mean(X | y=0). Returns w of shape (m,)."""
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    return mu_pos - mu_neg


def bias_weights_lda(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method 2: LDA direction (for binary y -> 1 component)."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    w = lda.scalings_[:, 0]
    # Orient so that (X_pos @ w).mean() > (X_neg @ w).mean()
    if X[y == 1].dot(w).mean() < X[y == 0].dot(w).mean():
        w = -w
    return w


def bias_weights_pls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method 3: first PLS component using PLSRegression with 1 component."""
    y_use = y.astype(float).reshape(-1, 1)
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X, y_use)
    w = pls.x_weights_[:, 0]
    if X[y == 1].dot(w).mean() < X[y == 0].dot(w).mean():
        w = -w
    return w


def bias_weights_logreg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method 4: logistic regression coefficients (positive -> predicts y=1).

    Features are standardized internally for numerical stability (LBFGS
    struggles with the scale disparity across emotion projections when N is
    large). Weights are then un-scaled so that ``X_raw @ w`` preserves the
    same ranking over examples as the fitted logistic model:

        logit_i = c @ z_i + b       where z_i = (x_i - mu) / sigma
                = (c / sigma) @ x_i + (b - (c / sigma) @ mu)
                =      w_raw @ x_i  +       const

    The constant is absorbed by the suggest_threshold step downstream.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=5000, random_state=42, solver="lbfgs")
    clf.fit(Xs, y)
    w_raw = clf.coef_[0] / scaler.scale_
    return w_raw.astype(np.float32)


BIAS_METHODS = {
    "diff":   bias_weights_diff,
    "lda":    bias_weights_lda,
    "pls":    bias_weights_pls,
    "logreg": bias_weights_logreg,
}


def weights_to_hidden_direction(w: np.ndarray, V: torch.Tensor) -> torch.Tensor:
    """Feature-space weights (m,) -> hidden-space direction V @ w of shape (d,)."""
    w_t = torch.from_numpy(np.asarray(w, dtype=np.float32))
    return V.float() @ w_t


# ============================================================
# Residual decomposition (Stage C)
# ============================================================

def decompose_against_subspace(
    d_bias: torch.Tensor,   # (d,)
    U: torch.Tensor,        # (d, k) orthonormal
) -> Dict[str, object]:
    """
    Decompose d_bias into protected part d_parallel (in span(U)) and residual d_perp.

    Returns:
        dict with d_parallel, d_perp (tensors) and norm / overlap scalars.
        overlap_ratio = ||d_parallel|| / ||d_bias||  in [0, 1]
    """
    d_bias = d_bias.float()
    U = U.float()
    coords = U.T @ d_bias          # (k,)
    d_parallel = U @ coords         # (d,)
    d_perp = d_bias - d_parallel    # (d,)
    norm_b = float(d_bias.norm().item())
    norm_p = float(d_parallel.norm().item())
    norm_r = float(d_perp.norm().item())
    overlap = norm_p / norm_b if norm_b > 1e-12 else 0.0
    return {
        "d_parallel": d_parallel,
        "d_perp": d_perp,
        "overlap_ratio": float(overlap),
        "norm_bias": norm_b,
        "norm_parallel": norm_p,
        "norm_perp": norm_r,
    }


# ============================================================
# Risk score for conditional steering
# ============================================================

def compute_risk_scores(
    X_features: np.ndarray,   # (N, m)
    w: np.ndarray,            # (m,) feature-space weights
) -> np.ndarray:
    """Risk score per example: r_i = w^T s_i. Higher = more bias-prone."""
    return X_features @ w


def suggest_threshold(
    risk_scores: np.ndarray,
    y: Optional[np.ndarray] = None,
    quantile: float = 0.5,
) -> float:
    """
    Suggest a threshold tau for conditional steering.
    If y is provided, return the median of r among bias-positive examples.
    Otherwise, return the given quantile of the full risk distribution.
    """
    if y is not None and (y == 1).any():
        return float(np.quantile(risk_scores[y == 1], 0.5))
    return float(np.quantile(risk_scores, quantile))
