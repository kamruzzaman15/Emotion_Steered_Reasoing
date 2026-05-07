"""
compute_mitigation_directions.py -- Stages B and C of the mitigation pipeline.

For each (dataset, bias-label definition, layer) we:
  B.1 Build the feature-space matrix X and label vector y from existing probe results.
  B.2 Fit four bias-direction methods in feature space:
        w_diff, w_lda, w_pls, w_logreg     (all shape (m,))
  B.3 Lift each to hidden-state space: d_bias = V @ w        (shape (d,))
  C   Decompose each d_bias against both protected subspaces (variance-based
      and task-validation-based), storing d_parallel, d_perp, and overlap ratio.

Sources used (all REUSED, nothing re-computed from scratch):
  - outputs/{slug}_emotion_vectors.pt       (for V, emotions order)
  - outputs/{slug}_protected_subspace.pt    (for U_variance, U_task)
  - outputs/{slug}_stereoset_results.pt     (for X, y for StereoSet)
  - outputs/{slug}_genassoc_results.pt      (for X, y for GenAssocBias)
  - outputs/{slug}_bbq_results.pt           (for X, y for BBQ-ambig)

Label definitions:
  - StereoSet / GenAssocBias:  y=1 iff prefers_stereotype==True (on the
       "stereotype"-condition projection; we use that condition's features
       because those are the activations the model saw while endorsing the
       stereotype).
  - BBQ-ambig: y=1 iff response_type == "stereotyped_guess",
               y=0 iff response_type == "correct",
               others are dropped. Features taken at the question position.
  - POOLED: concatenate all three sets above with a common feature ordering.

Output: outputs/{slug}_mitigation_directions.pt with nested dict
    scope                     -> one of {"stereoset", "genassoc", "bbq_ambig", "pooled"}
      layer_idx               -> int
        "n_train", "n_pos", "n_neg"  : sample counts
        "method"              -> one of {"diff", "lda", "pls", "logreg"}
          "w"                 : (m,) feature-space weights (numpy float32)
          "d_bias"            : (d,) hidden-space bias direction (torch float32)
          "risk_scores"       : (n_train,) r_i = w^T s_i (numpy float32)
          "tau_suggest"       : float, suggested threshold (median on y==1)
          "variance_subspace" :
              "d_parallel", "d_perp", "overlap_ratio", "norm_bias", "norm_perp"
          "task_subspace":
              "d_parallel", "d_perp", "overlap_ratio", "norm_bias", "norm_perp"
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from mitigation_utils import (  # noqa: E402
    build_V_matrix,
    BIAS_METHODS,
    weights_to_hidden_direction,
    decompose_against_subspace,
    compute_risk_scores,
    suggest_threshold,
)


# ============================================================
# Helpers: build (X, y) matrices from existing probe results
# ============================================================

def _stereoset_style_xy(
    results: List[dict],
    emotions_order: List[str],
    layer_idx: int,
    condition: str = "stereotype",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For StereoSet / GenAssocBias:
      X = emotion projections at `condition` (default "stereotype")
      y = 1 if prefers_stereotype else 0
    """
    X_rows, y_rows = [], []
    for r in results:
        proj_by_cond = r.get("emotion_projections", {})
        proj = proj_by_cond.get(condition, {}).get(layer_idx, {})
        if not proj or not all(e in proj for e in emotions_order):
            continue
        X_rows.append([proj[e] for e in emotions_order])
        y_rows.append(1 if r.get("prefers_stereotype") else 0)
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y


def _bbq_ambig_xy(
    results: List[dict],
    emotions_order: List[str],
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BBQ-ambig only:
      y=1 for "stereotyped_guess", y=0 for "correct", all others dropped.
      X = emotion_projections_at_question at the requested layer.
    """
    X_rows, y_rows = [], []
    for r in results:
        if r.get("condition") != "ambig":
            continue
        rt = r.get("response_type")
        if rt == "stereotyped_guess":
            label = 1
        elif rt == "correct":
            label = 0
        else:
            continue
        proj = r.get("emotion_projections_at_question", {}).get(layer_idx, {})
        if not proj or not all(e in proj for e in emotions_order):
            continue
        X_rows.append([proj[e] for e in emotions_order])
        y_rows.append(label)
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y


def _safe_load(path: Optional[str]):
    if path is None or not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


# ============================================================
# Main computation
# ============================================================

def compute_for_scope(
    scope: str,
    X: np.ndarray,
    y: np.ndarray,
    V: torch.Tensor,
    U_variance: torch.Tensor,
    U_task: torch.Tensor,
) -> dict:
    """
    Given feature matrix X (N, m), labels y (N,), emotion matrix V (d, m), and
    two protected bases, compute all four bias directions and their decompositions.
    """
    out = {
        "n_train": int(len(X)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "methods": {},
    }

    if len(X) < 10 or len(set(y.tolist())) < 2 or out["n_pos"] < 2 or out["n_neg"] < 2:
        out["skipped"] = "insufficient class variation"
        return out

    for name, fn in BIAS_METHODS.items():
        try:
            w = fn(X, y).astype(np.float32)
        except Exception as e:
            out["methods"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        d_bias = weights_to_hidden_direction(w, V)           # (d,) torch

        dec_var = decompose_against_subspace(d_bias, U_variance)
        dec_task = decompose_against_subspace(d_bias, U_task)

        r = compute_risk_scores(X, w)
        tau = suggest_threshold(r, y=y)

        out["methods"][name] = {
            "w": w,
            "d_bias": d_bias,
            "risk_scores": r.astype(np.float32),
            "tau_suggest": float(tau),
            "variance_subspace": {
                "d_parallel": dec_var["d_parallel"],
                "d_perp": dec_var["d_perp"],
                "overlap_ratio": dec_var["overlap_ratio"],
                "norm_bias": dec_var["norm_bias"],
                "norm_perp": dec_var["norm_perp"],
            },
            "task_subspace": {
                "d_parallel": dec_task["d_parallel"],
                "d_perp": dec_task["d_perp"],
                "overlap_ratio": dec_task["overlap_ratio"],
                "norm_bias": dec_task["norm_bias"],
                "norm_perp": dec_task["norm_perp"],
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion_vectors", type=str, required=True)
    parser.add_argument("--protected_subspace", type=str, required=True)
    parser.add_argument("--stereoset_results", type=str, default=None)
    parser.add_argument("--genassoc_results", type=str, default=None)
    parser.add_argument("--bbq_results", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layers", type=str, default="auto",
                        help="'auto' (use layers from protected subspace), 'all', or comma-separated list")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ---- Load reusable artifacts ----
    ev = torch.load(args.emotion_vectors, weights_only=False)
    emotions_order: List[str] = ev["emotions"]
    vectors_by_layer = ev["vectors"]

    subsp = torch.load(args.protected_subspace, weights_only=False)
    per_layer_subsp = subsp["per_layer"]
    all_sub_layers = sorted(per_layer_subsp.keys())

    # Resolve target layers (intersect with what we have subspace for)
    if args.layers == "auto":
        target_layers = list(subsp.get("target_layers", all_sub_layers))
    elif args.layers == "all":
        target_layers = all_sub_layers
    else:
        target_layers = [int(x) for x in args.layers.split(",")]
    target_layers = [l for l in target_layers if l in per_layer_subsp and l in vectors_by_layer]
    print(f"[directions] Target layers: {target_layers}")

    # ---- Load probe results (optional; skip scope if missing) ----
    ss = _safe_load(args.stereoset_results)
    ga = _safe_load(args.genassoc_results)
    bq = _safe_load(args.bbq_results)

    if ss is not None:
        print(f"[directions] StereoSet: {len(ss['results'])} examples")
    if ga is not None:
        print(f"[directions] GenAssocBias: {len(ga['results'])} examples")
    if bq is not None:
        print(f"[directions] BBQ: {len(bq['results'])} examples")

    output: Dict[str, Dict[int, dict]] = {
        "stereoset": {},
        "genassoc": {},
        "bbq_ambig": {},
        "pooled": {},
    }

    print("\n" + "=" * 92)
    print(f"{'scope':<12} | {'layer':>5} | {'method':<7} | {'N':>5} | {'+ve':>4} | "
          f"{'|d|':>8} | {'ovlp_var':>8} | {'ovlp_task':>9}")
    print("-" * 92)

    for l in target_layers:
        V = build_V_matrix(vectors_by_layer[l], emotions_order)  # (d, m)
        U_var = per_layer_subsp[l]["U_variance"]                  # (d, k_var)
        U_task = per_layer_subsp[l]["U_task"]                     # (d, k_task)

        pooled_X, pooled_y = [], []

        # ---- Per-dataset scopes ----
        scopes: List[Tuple[str, np.ndarray, np.ndarray]] = []

        if ss is not None:
            X, y = _stereoset_style_xy(ss["results"], emotions_order, l, "stereotype")
            scopes.append(("stereoset", X, y))
            if len(X) > 0:
                pooled_X.append(X); pooled_y.append(y)

        if ga is not None:
            X, y = _stereoset_style_xy(ga["results"], emotions_order, l, "stereotype")
            scopes.append(("genassoc", X, y))
            if len(X) > 0:
                pooled_X.append(X); pooled_y.append(y)

        if bq is not None:
            X, y = _bbq_ambig_xy(bq["results"], emotions_order, l)
            scopes.append(("bbq_ambig", X, y))
            if len(X) > 0:
                pooled_X.append(X); pooled_y.append(y)

        # ---- Pooled scope ----
        if pooled_X:
            pX = np.concatenate(pooled_X, axis=0)
            py = np.concatenate(pooled_y, axis=0)
            scopes.append(("pooled", pX, py))

        # ---- Compute for each scope ----
        for scope_name, X, y in scopes:
            if len(X) == 0:
                continue
            res = compute_for_scope(scope_name, X, y, V, U_var, U_task)
            output.setdefault(scope_name, {})[l] = res

            # Print summary
            for method_name, method_res in res.get("methods", {}).items():
                if "error" in method_res:
                    print(f"{scope_name:<12} | {l:>5d} | {method_name:<7} | "
                          f"{'-':>5} | {'-':>4} | ERROR: {method_res['error']}")
                    continue
                norm_b = method_res["variance_subspace"]["norm_bias"]
                ovlp_v = method_res["variance_subspace"]["overlap_ratio"]
                ovlp_t = method_res["task_subspace"]["overlap_ratio"]
                print(f"{scope_name:<12} | {l:>5d} | {method_name:<7} | "
                      f"{res['n_train']:>5d} | {res['n_pos']:>4d} | "
                      f"{norm_b:>8.3f} | {ovlp_v:>8.3f} | {ovlp_t:>9.3f}")
    print("=" * 92)

    # ---- Save ----
    payload = {
        "model_name": ev["model_name"],
        "emotions": emotions_order,
        "target_layers": target_layers,
        "scopes": output,
        "notes": {
            "label_defs": {
                "stereoset": "y=1 iff prefers_stereotype==True; X at 'stereotype' condition",
                "genassoc":  "y=1 iff prefers_stereotype==True; X at 'stereotype' condition",
                "bbq_ambig": "y=1 stereotyped_guess, y=0 correct (ambig only); X at question",
                "pooled":    "concatenation of all three above",
            },
        },
    }
    torch.save(payload, args.output)
    print(f"\n[directions] Saved mitigation directions to: {args.output}")


if __name__ == "__main__":
    main()
