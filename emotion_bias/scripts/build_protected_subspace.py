"""
build_protected_subspace.py -- Stage A of the mitigation pipeline.

Goal: identify, per layer, the smallest subspace of emotion-feature space that
still supports general (non-bias) emotion reasoning. That subspace is lifted
into hidden-state space via the existing emotion vectors and orthonormalized
via QR decomposition, yielding a basis U^(l) in R^d that we will protect from
intervention during steering.

What this script does:
  1. Load the existing per-layer emotion vectors (REUSE; error if missing).
  2. Load or build a cache of hidden states on the GoEmotions dataset
     (simplified config, train split, single-label examples only).
     - Hidden states are cached at the layer level as (N, d) float16 tensors
       so that re-running PCA / re-computing subspaces is fast and cheap.
  3. For each target layer:
       a. Project hidden states onto emotion vectors to form X_features (N, m).
       b. Fit PCA.
       c. Select k via two methods (reported side-by-side):
          - variance-based: smallest k with cumulative EVR >= gamma
          - task-validation:   smallest k with CV-accuracy >= gamma * full-space CV-accuracy
       d. Build orthonormal protected bases U_variance and U_task in R^d.
  4. Save everything to outputs/{slug}_protected_subspace.pt.

Only the "simplified" config's single-label examples are used, to give a clean
multi-class supervised signal for the task-validation selection of k.
"""

import os
import sys
import argparse
import time
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Local imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_utils import load_model, get_hidden_states_batch  # noqa: E402
from mitigation_utils import (  # noqa: E402
    build_V_matrix,
    hidden_to_features,
    fit_pca_on_features,
    select_k_variance,
    select_k_task_validation,
    build_protected_basis_in_hidden_space,
)


# ============================================================
# GoEmotions loader (robust to datasets library path changes)
# ============================================================

def load_goemotions_single_label(max_examples: int = 5000, seed: int = 42):
    """
    Load GoEmotions (simplified) train split, keep examples with exactly one label.
    Returns:
        texts: list[str] of length N
        labels: np.ndarray of int labels (0..27)
        label_names: list[str] length <= 28
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    # Try the canonical path first, then a fallback alias.
    last_err: Optional[Exception] = None
    for path in ["google-research-datasets/go_emotions", "go_emotions"]:
        try:
            ds = load_dataset(path, "simplified", split="train")
            label_names = ds.features["labels"].feature.names
            print(f"[goemotions] Loaded {len(ds)} examples from '{path}' (simplified/train)")
            break
        except Exception as e:
            last_err = e
            print(f"[goemotions] Failed to load from '{path}': {type(e).__name__}: {e}")
    else:
        raise RuntimeError(f"Could not load GoEmotions from any known path. Last error: {last_err}")

    # Single-label only (the dataset stores labels as a list of ints per row)
    texts, labels = [], []
    for row in ds:
        labs = row["labels"]
        if isinstance(labs, list) and len(labs) == 1:
            texts.append(row["text"])
            labels.append(labs[0])
    print(f"[goemotions] Single-label examples: {len(texts)}")

    # Subsample to max_examples (stratified by label when possible, else uniform)
    if max_examples and len(texts) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=max_examples, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"[goemotions] Subsampled to {len(texts)} examples (seed={seed})")

    return texts, np.array(labels, dtype=np.int64), label_names


# ============================================================
# Hidden-state extraction (cached per layer)
# ============================================================

def extract_hidden_states_for_texts(
    model,
    tokenizer,
    texts: List[str],
    target_layers: List[int],
    batch_size: int = 8,
) -> dict:
    """
    Run the model over each text and collect the last-token residual at each
    target layer. Returns a dict layer_idx -> torch.Tensor of shape (N, d) in
    float32 (CPU).
    """
    n = len(texts)
    d = model.config.hidden_size
    acc = {l: torch.empty(n, d, dtype=torch.float32) for l in target_layers}

    i = 0
    for start in tqdm(range(0, n, batch_size), desc="[goemotions] extracting hidden states"):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        per_ex = get_hidden_states_batch(
            model, tokenizer, batch_texts, layers=target_layers,
            token_position="last", batch_size=batch_size,
        )
        for item in per_ex:
            for l in target_layers:
                acc[l][i] = item[l].float().cpu()
            i += 1
    assert i == n, f"collected {i} examples, expected {n}"
    return acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model id, e.g. google/gemma-2-2b-it")
    parser.add_argument("--emotion_vectors", type=str, required=True,
                        help="Path to existing emotion_vectors.pt (will be loaded, not regenerated)")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to save the protected subspace (.pt)")
    parser.add_argument("--hidden_cache", type=str, default=None,
                        help="Optional path to cache/load GoEmotions hidden states. "
                             "If exists, it is loaded and the model forward pass is skipped.")
    parser.add_argument("--max_examples", type=int, default=5000,
                        help="Max number of GoEmotions single-label examples to use")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Preservation ratio for both variance-based and task-based k selection")
    parser.add_argument("--cv", type=int, default=3,
                        help="Cross-validation folds for task-based k selection")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layers", type=str, default="auto",
                        help="'auto' (use target_layers from emotion_vectors.pt), 'all', or comma-separated list")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ---- Load emotion vectors (REUSE) ----
    if not os.path.exists(args.emotion_vectors):
        raise FileNotFoundError(
            f"Emotion vectors not found at {args.emotion_vectors}. "
            f"Run scripts/extract_emotion_vectors.py first (or let run_all.sh build them)."
        )
    print(f"[build] Loading emotion vectors from: {args.emotion_vectors}")
    ev = torch.load(args.emotion_vectors, weights_only=False)
    emotions_order: List[str] = ev["emotions"]
    vectors_by_layer = ev["vectors"]
    hidden_dim = ev["hidden_dim"]
    num_layers = ev["num_layers"]
    model_name = ev["model_name"]
    all_layers = sorted(vectors_by_layer.keys())

    # Resolve target layers
    if args.layers == "auto":
        target_layers = list(ev.get("target_layers", all_layers))
    elif args.layers == "all":
        target_layers = all_layers
    else:
        target_layers = [int(x) for x in args.layers.split(",")]
    target_layers = [l for l in target_layers if l in vectors_by_layer]
    print(f"[build] Target layers: {target_layers}")
    print(f"[build] {len(emotions_order)} emotions, hidden_dim={hidden_dim}, num_layers={num_layers}")

    # ---- Load GoEmotions ----
    texts, y, label_names = load_goemotions_single_label(max_examples=args.max_examples)
    n_ex = len(texts)

    # ---- Extract or load hidden states ----
    hidden_cache_exists = args.hidden_cache is not None and os.path.exists(args.hidden_cache)
    if hidden_cache_exists:
        print(f"[build] Loading cached hidden states from: {args.hidden_cache}")
        cache = torch.load(args.hidden_cache, weights_only=False)
        # Sanity-check cache matches current request
        if cache.get("model_name") != model_name:
            print(f"  [warn] cache model_name ({cache.get('model_name')}) != current ({model_name}); ignoring cache")
            hidden_cache_exists = False
        elif cache.get("n_examples") != n_ex:
            print(f"  [warn] cache n_examples ({cache.get('n_examples')}) != current ({n_ex}); ignoring cache")
            hidden_cache_exists = False
        else:
            missing = [l for l in target_layers if l not in cache["hidden_states"]]
            if missing:
                print(f"  [warn] cache missing layers {missing}; re-extracting")
                hidden_cache_exists = False

    if not hidden_cache_exists:
        print(f"[build] Extracting hidden states for {n_ex} GoEmotions examples "
              f"at {len(target_layers)} layers...")
        model, tokenizer = load_model(args.model)
        t0 = time.time()
        hidden_states = extract_hidden_states_for_texts(
            model, tokenizer, texts, target_layers, batch_size=args.batch_size,
        )
        print(f"[build] Hidden-state extraction took {time.time() - t0:.1f}s")
        # Persist cache for future runs
        if args.hidden_cache is not None:
            os.makedirs(os.path.dirname(args.hidden_cache), exist_ok=True)
            torch.save({
                "model_name": model_name,
                "n_examples": n_ex,
                "target_layers": target_layers,
                "hidden_states": {l: hidden_states[l].to(torch.float16) for l in target_layers},
                "labels": y,
                "label_names": label_names,
            }, args.hidden_cache)
            print(f"[build] Cached hidden states to: {args.hidden_cache}")
        # Free GPU memory before heavy scikit-learn work
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        hidden_states = {l: cache["hidden_states"][l].float() for l in target_layers}
        y = cache["labels"]
        label_names = cache["label_names"]
        print(f"[build] Using cached hidden states for {n_ex} examples "
              f"at {len(target_layers)} layers")

    # ---- Per-layer protected subspace construction ----
    per_layer = {}
    print("\n" + "=" * 72)
    print(f"{'Layer':>6} | {'m':>3} | {'k_var':>5} | {'k_task':>6} | "
          f"{'acc_full':>8} | {'acc_k_var':>9} | {'acc_k_task':>10}")
    print("-" * 72)
    for l in target_layers:
        V = build_V_matrix(vectors_by_layer[l], emotions_order)  # (d, m)
        H = hidden_states[l].float()                              # (N, d)
        X = hidden_to_features(H, V).numpy()                      # (N, m)

        pca, scaler = fit_pca_on_features(X, standardize=True)
        k_var = select_k_variance(pca, gamma=args.gamma)
        k_task, per_k = select_k_task_validation(
            X, y, pca=pca, scaler=scaler, gamma=args.gamma, cv=args.cv,
        )

        U_var = build_protected_basis_in_hidden_space(V, pca.components_, k_var)
        U_task = build_protected_basis_in_hidden_space(V, pca.components_, k_task)

        per_layer[l] = {
            "V": V,                             # (d, m) -- stored so downstream scripts can reuse
            "pca_components": pca.components_,  # (m, m)
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            "scaler_mean": scaler.mean_ if scaler is not None else None,
            "scaler_scale": scaler.scale_ if scaler is not None else None,
            "k_variance": k_var,
            "k_task": k_task,
            "task_acc_full": per_k["full"],
            "task_acc_per_k": {kk: vv for kk, vv in per_k.items() if kk != "full"},
            "U_variance": U_var,                # (d, k_var) orthonormal
            "U_task": U_task,                   # (d, k_task) orthonormal
        }

        acc_full = per_k["full"]
        acc_kv = per_k.get(k_var, float("nan"))
        acc_kt = per_k.get(k_task, float("nan"))
        print(f"{l:>6d} | {len(emotions_order):>3d} | {k_var:>5d} | {k_task:>6d} | "
              f"{acc_full:>8.3f} | {acc_kv:>9.3f} | {acc_kt:>10.3f}")
    print("=" * 72)

    # ---- Save ----
    out = {
        "model_name": model_name,
        "emotions": emotions_order,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "target_layers": target_layers,
        "gamma": args.gamma,
        "cv": args.cv,
        "n_task_examples": n_ex,
        "label_names": label_names,
        "per_layer": per_layer,
    }
    torch.save(out, args.output)
    print(f"\n[build] Saved protected subspace to: {args.output}")
    print(f"        Per-layer k_variance / k_task stored under per_layer[layer_idx]")


if __name__ == "__main__":
    main()
