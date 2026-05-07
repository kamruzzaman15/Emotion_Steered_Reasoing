"""
probe_with_steering.py -- Stage D of the mitigation pipeline.

Re-run the existing probes (StereoSet, GenAssocBias, BBQ) with activation
steering applied via a forward hook on the chosen layer(s). The residual
stream at each steered layer is modified as:

      h <- h - alpha * d_perp          (unconditional)
   or h <- h - alpha * d_perp * (r > tau)  (conditional, using risk score)

where d_perp is the residual-after-protected-subspace bias direction computed
in Stage B+C, and r = w^T (V^T h_last_token) uses the same feature-space
weights w (stored alongside d_perp).

This script imports the single-example probe functions from your existing
probe_stereoset.py / probe_genassocbias.py / probe_bbq.py (nothing duplicated),
registers a hook, and runs them through. Output format matches the original
probes exactly, so analyze_results.py is a drop-in.

Design choices (Gene's answers):
  - Single-layer first, multi-layer as ablation (--target_layers supports both).
  - Choice of bias-direction method is a CLI flag (--method ∈ {diff,lda,pls,logreg}).
  - Choice of protected subspace (--protected_subspace ∈ {variance, task}).
  - Choice of scope (which bias-direction was fit): --scope ∈ {stereoset, genassoc, bbq_ambig, pooled}.
  - --conditional flag enables risk-gated steering.
"""

import os
import sys
import json
import argparse
from contextlib import contextmanager
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_utils import load_model  # noqa: E402
from mitigation_utils import build_V_matrix  # noqa: E402

# Existing probe functions (reused unchanged)
from probe_stereoset import probe_single_example as probe_stereo  # noqa: E402
from probe_genassocbias import probe_single_example as probe_genassoc  # noqa: E402
from probe_bbq import probe_single_example as probe_bbq  # noqa: E402


# ============================================================
# Forward-hook steering
# ============================================================

class SteeringHook:
    """
    Forward-hook wrapper that subtracts alpha * d_perp from the residual stream
    at the output of one or more decoder layers. Supports unconditional and
    conditional (risk-gated) modes.

    For a given target layer l:
        output = layer(input)     # hidden_states tensor or tuple
        h = output if Tensor else output[0]     # shape (batch, seq_len, d)

        unconditional:
            h_new = h - alpha * d_perp

        conditional:
            r = (V^T h[:, -1, :]) @ w                       # (batch,)
            mask = (r > tau).float().view(-1, 1, 1)         # (batch, 1, 1)
            h_new = h - alpha * d_perp * mask

    The hook must be registered on each target layer. Emotion vectors V and
    weights w are needed only for conditional mode.
    """

    def __init__(
        self,
        model,
        target_layers: List[int],
        d_perp_per_layer: dict,       # layer_idx -> tensor (d,)
        alpha: float,
        V_per_layer: Optional[dict] = None,   # layer_idx -> (d, m); required if conditional
        w_per_layer: Optional[dict] = None,   # layer_idx -> (m,);   required if conditional
        tau_per_layer: Optional[dict] = None, # layer_idx -> float;  required if conditional
        conditional: bool = False,
    ):
        self.model = model
        self.target_layers = list(target_layers)
        self.alpha = float(alpha)
        self.conditional = conditional

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Pre-move steering tensors to the model's device/dtype
        self.d_perp_per_layer = {
            l: d_perp_per_layer[l].to(device=device, dtype=dtype).contiguous()
            for l in self.target_layers
        }

        if conditional:
            assert V_per_layer is not None and w_per_layer is not None and tau_per_layer is not None, (
                "Conditional steering requires V_per_layer, w_per_layer, tau_per_layer."
            )
            self.V_per_layer = {
                l: V_per_layer[l].to(device=device, dtype=dtype).contiguous()
                for l in self.target_layers
            }
            self.w_per_layer = {
                l: torch.as_tensor(w_per_layer[l]).to(device=device, dtype=dtype).contiguous()
                for l in self.target_layers
            }
            self.tau_per_layer = {l: float(tau_per_layer[l]) for l in self.target_layers}

        self._handles = []

        # Instrumentation for debugging hook behavior. Tracks per-layer:
        #   - call_count: how many times the hook fired
        #   - output_was_tuple: whether the layer output was a tuple
        #   - h_norm_mean_before / h_norm_mean_after: pre/post perturbation norms
        #   - tuple_first_call_logged: ensures we only print once per layer
        self._diag = {
            l: {"call_count": 0, "output_was_tuple": None,
                "h_norm_sum_before": 0.0, "h_norm_sum_after": 0.0,
                "n_tokens": 0, "first_call_logged": False}
            for l in self.target_layers
        }
        # Print up to this many early hook firings per layer
        self._max_early_prints = 3

    def _resolve_decoder_layers(self):
        """Return the module list holding the decoder layers (model.model.layers for
        Gemma-2 / Llama / Mistral / Qwen / Phi / OLMo)."""
        m = self.model
        for path in [("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")]:
            obj = m
            ok = True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                return obj
        raise RuntimeError("Could not resolve decoder-layer module list on this model.")

    def _make_hook_fn(self, layer_idx: int):
        d_perp = self.d_perp_per_layer[layer_idx]  # (d,)
        alpha = self.alpha
        conditional = self.conditional
        if conditional:
            V = self.V_per_layer[layer_idx]        # (d, m)
            w = self.w_per_layer[layer_idx]        # (m,)
            tau = self.tau_per_layer[layer_idx]

        # Precompute broadcasting shape (1, 1, d)
        d_perp_bc = d_perp.view(1, 1, -1)

        diag = self._diag[layer_idx]
        max_early = self._max_early_prints

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
                was_tuple = True
            else:
                h = output
                rest = None
                was_tuple = False

            # --- Compute the perturbation ---
            if conditional:
                last = h[:, -1, :]                    # (batch, d)
                s = last @ V                          # (batch, m)
                r = s @ w                             # (batch,)
                mask = (r > tau).to(h.dtype).view(-1, 1, 1)
                h_new = h - alpha * d_perp_bc * mask
            else:
                h_new = h - alpha * d_perp_bc

            # --- Instrumentation: track norms + print first few calls ---
            with torch.no_grad():
                # h has shape (batch, seq_len, d); take token-wise norms
                norms_before = h.float().norm(dim=-1)      # (batch, seq_len)
                norms_after = h_new.float().norm(dim=-1)   # (batch, seq_len)
                diag["h_norm_sum_before"] += float(norms_before.sum().item())
                diag["h_norm_sum_after"] += float(norms_after.sum().item())
                diag["n_tokens"] += int(norms_before.numel())
                diag["output_was_tuple"] = was_tuple
                diag["call_count"] += 1

                if diag["call_count"] <= max_early:
                    delta_norm = float((h_new - h).float().norm(dim=-1).mean().item())
                    print(
                        f"[HOOK] layer {layer_idx} call#{diag['call_count']}: "
                        f"h.shape={tuple(h.shape)}  "
                        f"||h||_mean_before={float(norms_before.mean().item()):.3f}  "
                        f"||h||_mean_after={float(norms_after.mean().item()):.3f}  "
                        f"||Δh||_mean={delta_norm:.3f}  "
                        f"output_was_tuple={was_tuple}  "
                        f"dtype={h.dtype}",
                        flush=True,
                    )

            if rest is None:
                return h_new
            return (h_new,) + rest

        return hook

    def __enter__(self):
        layers_module = self._resolve_decoder_layers()
        print(f"[HOOK] Registering forward hooks on {len(self.target_layers)} "
              f"layer(s): {self.target_layers}", flush=True)
        for l in self.target_layers:
            handle = layers_module[l].register_forward_hook(self._make_hook_fn(l))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

        # Print a summary of hook activity for debugging.
        print("\n" + "=" * 72)
        print("[HOOK] FINAL DIAGNOSTIC SUMMARY")
        print("=" * 72)
        for l in self.target_layers:
            d = self._diag[l]
            n = d["call_count"]
            if n == 0:
                print(f"  layer {l}: HOOK NEVER FIRED (call_count=0). "
                      f"Layer indexing is wrong OR hook did not register.")
                continue
            if d["n_tokens"] == 0:
                print(f"  layer {l}: {n} calls but zero tokens processed?")
                continue
            mean_before = d["h_norm_sum_before"] / d["n_tokens"]
            mean_after = d["h_norm_sum_after"] / d["n_tokens"]
            print(f"  layer {l}: calls={n}  tokens_seen={d['n_tokens']}  "
                  f"output_was_tuple={d['output_was_tuple']}")
            print(f"            mean ||h|| before hook = {mean_before:.3f}")
            print(f"            mean ||h|| after  hook = {mean_after:.3f}")
            print(f"            mean change in norm    = {mean_after - mean_before:+.3f}")
            if abs(mean_after - mean_before) < 1e-4:
                print(f"            WARNING: norms unchanged. The perturbation is")
                print(f"            either not applied, or the modified tensor is not")
                print(f"            being propagated downstream.")
        print("=" * 72, flush=True)

        return False


# ============================================================
# Main
# ============================================================

def _load_bbq_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--emotion_vectors", type=str, required=True)
    parser.add_argument("--mitigation_directions", type=str, required=True)
    parser.add_argument("--protected_subspace", type=str, required=True,
                        help="Only used to resolve which subspace's d_perp we pick from "
                             "the stored directions; the tensors are already in the "
                             "directions file. This arg is retained for traceability/logging.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["stereoset", "genassoc", "bbq"])
    parser.add_argument("--data", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output", type=str, required=True)

    # Steering configuration
    parser.add_argument("--scope", type=str, default="pooled",
                        choices=["stereoset", "genassoc", "bbq_ambig", "pooled"],
                        help="Which scope's bias direction to apply")
    parser.add_argument("--method", type=str, default="logreg",
                        choices=["diff", "lda", "pls", "logreg"])
    parser.add_argument("--subspace", type=str, default="task",
                        choices=["variance", "task"],
                        help="Which protected subspace's d_perp to use")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    parser.add_argument("--conditional", action="store_true",
                        help="Only steer when risk score r > tau")
    parser.add_argument("--tau", type=float, default=None,
                        help="Override tau for conditional steering. Defaults to stored tau_suggest.")
    parser.add_argument("--target_layers", type=str, required=True,
                        help="Comma-separated layer indices to steer at (e.g. '15' or '10,15,20')")

    # Probe parameters
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=12,
                        help="Only used for BBQ")
    parser.add_argument("--probe_layers", type=str, default="auto",
                        help="Which layers the probe should record activations at "
                             "(independent of steering layers). 'auto' uses target_layers "
                             "from emotion_vectors.pt")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ---- Resolve steering tensors ----
    ev = torch.load(args.emotion_vectors, weights_only=False)
    emotions_order = ev["emotions"]
    vectors_by_layer = ev["vectors"]

    mit = torch.load(args.mitigation_directions, weights_only=False)
    scope_data = mit["scopes"][args.scope]
    available_layers = sorted(scope_data.keys())
    if not available_layers:
        raise ValueError(
            f"Scope '{args.scope}' has no populated layers in directions file."
        )

    requested_layers = [int(x) for x in args.target_layers.split(",")]

    def _snap_to_nearest(l: int, available: list) -> int:
        """
        Return the available layer closest to l. Ties are broken toward the
        deeper layer (which generally encodes more task-specific / decision-
        relevant signal in causal LMs). This lets callers ask for, e.g.,
        layer 17 when the probe set has [16, 20] and get 16 (closer) rather
        than erroring; if they ask for 18, they get 20 (tie-break deeper).
        """
        diffs = [(abs(l - a), -a, a) for a in available]  # -a for deeper tiebreak
        diffs.sort()
        return diffs[0][2]

    target_layers = []
    for l in requested_layers:
        if l in scope_data:
            target_layers.append(l)
        else:
            snapped = _snap_to_nearest(l, available_layers)
            print(f"[warn] Layer {l} not available for scope='{args.scope}'. "
                  f"Snapping to nearest available layer: {snapped}. "
                  f"(Available: {available_layers})")
            target_layers.append(snapped)

    # Deduplicate while preserving order, in case multiple requested layers
    # snap to the same target.
    seen = set()
    target_layers = [l for l in target_layers if not (l in seen or seen.add(l))]

    for l in target_layers:
        if args.method not in scope_data[l].get("methods", {}):
            raise ValueError(
                f"Method '{args.method}' not found for scope='{args.scope}' at layer {l}. "
                f"Available methods: {list(scope_data[l].get('methods', {}).keys())}"
            )

    sub_key = f"{args.subspace}_subspace"  # "variance_subspace" or "task_subspace"

    d_perp_per_layer = {}
    V_per_layer = {}
    w_per_layer = {}
    tau_per_layer = {}
    for l in target_layers:
        m = scope_data[l]["methods"][args.method]
        d_perp_per_layer[l] = m[sub_key]["d_perp"]                             # (d,)
        V_per_layer[l] = build_V_matrix(vectors_by_layer[l], emotions_order)   # (d, m)
        w_per_layer[l] = m["w"]                                                # (m,)
        tau_per_layer[l] = args.tau if args.tau is not None else m["tau_suggest"]

    print(f"[steer] Scope: {args.scope}   Method: {args.method}   Subspace: {args.subspace}")
    print(f"[steer] Steering layer(s): {target_layers}   alpha={args.alpha}   "
          f"conditional={args.conditional}")
    for l in target_layers:
        norm_bias = scope_data[l]["methods"][args.method][sub_key]["norm_bias"]
        norm_perp = scope_data[l]["methods"][args.method][sub_key]["norm_perp"]
        overlap = scope_data[l]["methods"][args.method][sub_key]["overlap_ratio"]
        print(f"        layer {l}: ||d_bias||={norm_bias:.3f}  ||d_perp||={norm_perp:.3f}  "
              f"overlap={overlap:.3f}  tau={tau_per_layer[l]:.3f}")

    # ---- Resolve probe_layers (what to record) ----
    if args.probe_layers == "auto":
        probe_layers = sorted(vectors_by_layer.keys())
        # Match the 'quarter' heuristic used in the baseline probes
        num_layers = ev["num_layers"]
        probe_layers = [l for l in probe_layers
                        if (l % max(1, num_layers // 6) == 0) or l == num_layers - 1]
    else:
        probe_layers = [int(x) for x in args.probe_layers.split(",")]
    probe_layers = [l for l in probe_layers if l in vectors_by_layer]
    print(f"[steer] Recording activations at probe layers: {probe_layers}")

    # ---- Load model and data ----
    model, tokenizer = load_model(args.model)

    # Each branch mirrors the *baseline* probe's loading + subsampling order
    # EXACTLY so that example IDs line up between baseline and steered runs.
    # This is required for the McNemar paired test in analyze_mitigation.py.
    if args.dataset == "stereoset":
        # probe_stereoset.py: dataset = dataset[:args.max_examples]
        with open(args.data) as f:
            dataset = json.load(f)
        if args.max_examples and args.max_examples > 0:
            dataset = dataset[:args.max_examples]

    elif args.dataset == "genassoc":
        # probe_genassocbias.py: df.sample(..., random_state=42) THEN enumerate to set id
        df = pd.read_csv(args.data)
        if args.max_examples > 0 and len(df) > args.max_examples:
            df = df.sample(n=args.max_examples, random_state=42)
        dataset = df.to_dict(orient="records")
        for i, ex in enumerate(dataset):
            ex["id"] = f"genassoc_{i}"

    elif args.dataset == "bbq":
        # probe_bbq.py: random.sample with seed=42
        dataset = _load_bbq_jsonl(args.data)
        if args.max_examples > 0 and len(dataset) > args.max_examples:
            import random
            random.seed(42)
            dataset = random.sample(dataset, args.max_examples)

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    print(f"[steer] Dataset: {args.dataset}   N={len(dataset)}")

    # ---- Run with steering ----
    results = []
    with SteeringHook(
        model=model,
        target_layers=target_layers,
        d_perp_per_layer=d_perp_per_layer,
        alpha=args.alpha,
        V_per_layer=V_per_layer,
        w_per_layer=w_per_layer,
        tau_per_layer=tau_per_layer,
        conditional=args.conditional,
    ):
        for ex in tqdm(dataset, desc=f"[steer] Probing {args.dataset}"):
            try:
                if args.dataset == "stereoset":
                    r = probe_stereo(model, tokenizer, ex, vectors_by_layer, probe_layers)
                elif args.dataset == "genassoc":
                    r = probe_genassoc(model, tokenizer, ex, vectors_by_layer, probe_layers)
                elif args.dataset == "bbq":
                    r = probe_bbq(
                        model, tokenizer, ex, vectors_by_layer, probe_layers,
                        max_new_tokens=args.max_new_tokens,
                    )
                results.append(r)
            except Exception as e:
                print(f"  [warn] error on example {ex.get('id', ex.get('example_id', '?'))}: "
                      f"{type(e).__name__}: {e}")
                continue

    # ---- Save in the same format as baseline probes ----
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "target_layers": probe_layers,
        "emotions": emotions_order,
        "results": results,
        # Extra metadata specific to steered runs
        "steering": {
            "scope": args.scope,
            "method": args.method,
            "subspace": args.subspace,
            "steering_layers": target_layers,
            "alpha": args.alpha,
            "conditional": args.conditional,
            "tau_per_layer": tau_per_layer,
        },
    }
    torch.save(payload, args.output)
    print(f"\n[steer] Saved steered probe results to: {args.output}")

    # ---- Quick summary ----
    if args.dataset in ("stereoset", "genassoc"):
        n = len(results)
        n_s = sum(1 for r in results if r.get("prefers_stereotype"))
        print(f"[steer] Prefers stereotype: {n_s}/{n} ({100*n_s/max(1,n):.1f}%)")
    elif args.dataset == "bbq":
        amb = [r for r in results if r.get("condition") == "ambig"]
        n_sg = sum(1 for r in amb if r.get("response_type") == "stereotyped_guess")
        n_co = sum(1 for r in amb if r.get("response_type") == "correct")
        print(f"[steer] BBQ-ambig: correct={n_co}, stereotyped_guess={n_sg}, total={len(amb)}")


if __name__ == "__main__":
    main()
