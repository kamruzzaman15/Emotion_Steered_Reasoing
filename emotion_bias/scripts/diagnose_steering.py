"""
diagnose_steering.py -- Quick diagnostic for a single steered probe run.

Answers three questions given one baseline + one steered result file:
  1. Did ANY responses change under steering? (flip counts, b01/b10)
  2. Is the change directional (reducing bias) or noisy?
  3. Did emotion activations measurably shift at the steering layer?

If flip count is tiny, alpha is almost certainly too small. If the flips are
balanced, the direction may be wrong. The script also prints a suggested alpha
range based on the observed d_perp magnitude.

Usage:
    python scripts/diagnose_steering.py \
        --baseline outputs/<slug>_stereoset_results.pt \
        --steered  outputs/<slug>_stereoset_steered.pt \
        --directions outputs/<slug>_mitigation_directions.pt
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _id(r):
    return r.get("id", r.get("example_id"))


def _is_stereoset_style(result_list: List[dict]) -> bool:
    """StereoSet / GenAssocBias both store prefers_stereotype."""
    return len(result_list) > 0 and "prefers_stereotype" in result_list[0]


def _is_bbq(result_list: List[dict]) -> bool:
    return len(result_list) > 0 and "response_type" in result_list[0]


def compute_flip_stats(base_res: List[dict], steer_res: List[dict]):
    """
    Return dict with per-id pairing of binary 'bias-positive' outcomes.
    For StereoSet/GenAssoc: outcome = prefers_stereotype
    For BBQ: outcome = 1 if stereotyped_guess, 0 if correct, else skipped
    """
    b_map = {_id(r): r for r in base_res}
    s_map = {_id(r): r for r in steer_res}
    common = [k for k in b_map if k in s_map]

    both_outcomes = []  # list of (base_y, steer_y, base_resp, steer_resp)

    stereoset_style = _is_stereoset_style(base_res)

    for k in common:
        rb, rs = b_map[k], s_map[k]
        if stereoset_style:
            yb = 1 if rb.get("prefers_stereotype") else 0
            ys = 1 if rs.get("prefers_stereotype") else 0
            both_outcomes.append((yb, ys, rb, rs))
        else:  # BBQ
            if rb.get("condition") != "ambig" or rs.get("condition") != "ambig":
                continue
            rtb, rts = rb.get("response_type"), rs.get("response_type")
            mb = {"stereotyped_guess": 1, "correct": 0}.get(rtb)
            ms = {"stereotyped_guess": 1, "correct": 0}.get(rts)
            if mb is None or ms is None:
                continue
            both_outcomes.append((mb, ms, rb, rs))

    if not both_outcomes:
        return None

    yb = np.array([o[0] for o in both_outcomes])
    ys = np.array([o[1] for o in both_outcomes])
    n = len(yb)
    unchanged = int(np.sum(yb == ys))
    b01_worse = int(np.sum((yb == 0) & (ys == 1)))  # baseline clean -> steered biased
    b10_better = int(np.sum((yb == 1) & (ys == 0)))  # baseline biased -> steered clean

    # Exact two-sided binomial (McNemar) on discordant pairs
    n_disc = b01_worse + b10_better
    if n_disc == 0:
        p_value = 1.0
    else:
        k = min(b01_worse, b10_better)
        try:
            p_value = float(stats.binomtest(k, n=n_disc, p=0.5, alternative="two-sided").pvalue)
        except AttributeError:
            p_value = float(stats.binom_test(k, n=n_disc, p=0.5, alternative="two-sided"))

    return {
        "n_paired": n,
        "unchanged": unchanged,
        "flipped": n - unchanged,
        "b01_worse": b01_worse,
        "b10_better": b10_better,
        "baseline_rate": float(yb.mean()),
        "steered_rate": float(ys.mean()),
        "delta_rate": float(ys.mean() - yb.mean()),
        "p_value": p_value,
    }


def emotion_projection_shift(
    base_res: List[dict], steer_res: List[dict], layer: int
) -> Dict[str, float]:
    """
    Mean |steered_proj - baseline_proj| per emotion at the given layer.
    Uses whichever projection key the results have (stereoset-style vs bbq).
    """
    b_map = {_id(r): r for r in base_res}
    s_map = {_id(r): r for r in steer_res}
    common = [k for k in b_map if k in s_map]

    if len(common) == 0:
        return {}

    # Determine schema
    sample = b_map[common[0]]
    if "emotion_projections" in sample:
        # stereoset / genassoc - use the 'stereotype' condition
        def getp(r):
            return r.get("emotion_projections", {}).get("stereotype", {}).get(layer, {})
    elif "emotion_projections_at_question" in sample:
        def getp(r):
            return r.get("emotion_projections_at_question", {}).get(layer, {})
    else:
        return {}

    all_emotions = set()
    for r in base_res:
        all_emotions.update(getp(r).keys())
    for r in steer_res:
        all_emotions.update(getp(r).keys())
    if not all_emotions:
        return {}

    shifts = {e: [] for e in all_emotions}
    for k in common:
        pb, ps = getp(b_map[k]), getp(s_map[k])
        for e in all_emotions:
            if e in pb and e in ps:
                shifts[e].append(ps[e] - pb[e])

    return {e: float(np.mean(v)) if v else float("nan") for e, v in shifts.items()}


def pick_steering_info(steer_data: dict, directions_data: Optional[dict]):
    """Recover scope, method, subspace, steering_layers from steered file's metadata."""
    if "steering" not in steer_data:
        return None
    s = steer_data["steering"]
    layer = s.get("steering_layers", [None])[0]
    scope = s.get("scope")
    method = s.get("method")
    subspace = s.get("subspace")
    alpha = s.get("alpha")

    info = {"scope": scope, "method": method, "subspace": subspace,
            "alpha": alpha, "layer": layer}

    if directions_data is not None and layer is not None:
        try:
            entry = directions_data["scopes"][scope][layer]["methods"][method]
            key = f"{subspace}_subspace"
            info["norm_bias"] = entry[key]["norm_bias"]
            info["norm_perp"] = entry[key]["norm_perp"]
            info["overlap_ratio"] = entry[key]["overlap_ratio"]
            info["tau_suggest"] = entry["tau_suggest"]
            # Also pick up the weight-vector norm
            info["norm_w"] = float(np.linalg.norm(entry["w"]))
        except Exception as e:
            info["directions_lookup_error"] = str(e)

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--steered", type=str, required=True)
    parser.add_argument("--directions", type=str, default=None,
                        help="Optional: mitigation_directions.pt, used to look up "
                             "d_bias / d_perp norms at the steering layer")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Baseline : {args.baseline}")
    print(f"Steered  : {args.steered}")
    print("=" * 78)

    base = torch.load(args.baseline, weights_only=False)
    steer = torch.load(args.steered, weights_only=False)
    directions = torch.load(args.directions, weights_only=False) if args.directions else None

    base_res = base["results"]
    steer_res = steer["results"]
    print(f"Baseline N = {len(base_res)}   Steered N = {len(steer_res)}")

    # ---- Steering config + direction magnitudes ----
    info = pick_steering_info(steer, directions)
    if info is not None:
        print("\n[config] scope={scope}  method={method}  subspace={subspace}  "
              "alpha={alpha}  layer={layer}".format(**info))
        if "norm_bias" in info:
            print(f"[config] ||d_bias||={info['norm_bias']:.3f}   "
                  f"||d_perp||={info['norm_perp']:.3f}   "
                  f"overlap={info['overlap_ratio']:.3f}")
            effective_perturb = info["alpha"] * info["norm_perp"]
            print(f"[config] alpha * ||d_perp|| = {effective_perturb:.3f}  "
                  f"(per-token perturbation applied to residual stream)")

    # ---- Flip analysis ----
    print("\n" + "-" * 78)
    print("FLIP ANALYSIS")
    print("-" * 78)
    fs = compute_flip_stats(base_res, steer_res)
    if fs is None:
        print("  No paired outcomes available.")
    else:
        print(f"  Paired examples:     {fs['n_paired']}")
        print(f"  Unchanged:           {fs['unchanged']} "
              f"({100*fs['unchanged']/fs['n_paired']:.1f}%)")
        print(f"  Flipped total:       {fs['flipped']} "
              f"({100*fs['flipped']/fs['n_paired']:.1f}%)")
        print(f"    b10 (better):      {fs['b10_better']}    [baseline biased -> steered clean]")
        print(f"    b01 (worse):       {fs['b01_worse']}    [baseline clean -> steered biased]")
        print(f"  Baseline rate:       {fs['baseline_rate']:.4f}")
        print(f"  Steered rate:        {fs['steered_rate']:.4f}")
        print(f"  Delta:               {fs['delta_rate']:+.4f}")
        print(f"  McNemar exact p:     {fs['p_value']:.4g}")

        # Verdict
        print("\n  VERDICT:")
        if fs["flipped"] == 0:
            print("    NO RESPONSES CHANGED. alpha is far too small -- the perturbation")
            print("    is below Gemma-2-2B's behavioral threshold. Raise alpha by 5-50x.")
        elif fs["flipped"] < 10:
            print("    Almost no responses changed. alpha is too small; try 5x current.")
        elif fs["b10_better"] > 2 * fs["b01_worse"] and fs["p_value"] < 0.05:
            print("    Steering is directional and significantly reduces bias.")
        elif fs["b01_worse"] > 2 * fs["b10_better"] and fs["p_value"] < 0.05:
            print("    WARNING: steering is INCREASING bias (direction flipped?).")
            print("    Double-check sign convention in the bias-direction method.")
        elif abs(fs["b10_better"] - fs["b01_worse"]) < 0.2 * fs["flipped"]:
            print("    Steering causes symmetric flips -- disrupting responses but not")
            print("    consistently reducing bias. Direction may be off-target, or the")
            print("    layer choice isn't where this bias signal lives. Try other layers")
            print("    or methods.")
        else:
            print(f"    Modest directional effect (p={fs['p_value']:.3g}). "
                  f"Consider higher alpha or additional layers.")

    # ---- Emotion projection shift ----
    if info is not None and "layer" in info and info["layer"] is not None:
        # Use the analysis layer (2/3 depth) from the probe's target_layers,
        # matching analyze_results.py convention.
        layers = base.get("target_layers", [])
        if layers:
            analysis_layer = layers[len(layers) * 2 // 3]
        else:
            analysis_layer = info["layer"]
        print("\n" + "-" * 78)
        print(f"EMOTION PROJECTION SHIFT @ layer {analysis_layer} (steered - baseline)")
        print("-" * 78)
        shifts = emotion_projection_shift(base_res, steer_res, analysis_layer)
        if not shifts:
            print("  (no projections found at this layer)")
        else:
            sorted_shifts = sorted(shifts.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top = sorted_shifts[:10]
            max_abs = max(abs(v) for v in shifts.values()) if shifts else 0.0
            print(f"  Top 10 emotions by |shift|:")
            for e, v in top:
                bar = "+" if v > 0 else "-"
                print(f"    {e:<20}  {v:+.4f}")
            print(f"\n  Max |shift|: {max_abs:.4f}")
            if max_abs < 0.01:
                print("  -> Projections barely moved. Steering had almost no effect on")
                print("     downstream activations. alpha is too small.")
            elif max_abs < 0.1:
                print("  -> Small but measurable effect. alpha may need to be larger.")
            else:
                print("  -> Clear effect on activations. If behavior still didn't change,")
                print("     the direction may be off-target (wrong layer / wrong method).")

    # ---- Alpha suggestion ----
    if info is not None and "norm_perp" in info and fs is not None:
        print("\n" + "-" * 78)
        print("ALPHA TUNING SUGGESTION")
        print("-" * 78)
        current = info["alpha"]
        perturb = current * info["norm_perp"]
        print(f"  Current alpha = {current}, perturb_magnitude = alpha * ||d_perp|| = {perturb:.3f}")
        print(f"  Typical Gemma-2-2B hidden-state norm is roughly 20-100 at mid layers.")
        print(f"  A useful steering perturbation is ~5-20% of ||h||, so ~1-20 in norm.")
        target_perturbs = [1.0, 3.0, 10.0, 30.0]
        suggested = [round(tp / info["norm_perp"], 1) for tp in target_perturbs if info["norm_perp"] > 1e-6]
        if suggested:
            print(f"  Suggested alpha values to sweep (targeting perturbations "
                  f"{target_perturbs}):  {suggested}")

    print("=" * 78)


if __name__ == "__main__":
    main()
