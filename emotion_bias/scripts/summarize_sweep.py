"""
summarize_sweep.py  -- Rich before/after summary with full-config cache keys
                        and subset-aware baseline matching.

Prints a structured report containing:
  * Full steering configuration and all prereq paths
  * Direction magnitudes at the steering layer (||d_bias||, ||d_perp||, overlap)
  * Hidden-state scale at the steering layer (from GoEmotions cache, if present)
  * Per-dataset:
      - Baseline rate on the SAME IDs as the steered run  (paired)
      - Steered rate
      - Delta, flip counts (b10 better / b01 worse), McNemar p-value
      - Verdict (directional / symmetric / weak / none)
  * MMLU baseline vs steered accuracy, absolute & relative delta
  * Overall verdict that combines bias reduction and capability cost

The filename scheme it consumes matches run_mitigation_sweep.sh:
  outputs/<slug>_{ds}_steered_L{layer}_a{alpha}_{scope}_{method}_{subspace}_{cond}_N{maxex}.pt
  outputs/<slug>_mmlu_L{layer}_a{alpha}_{scope}_{method}_{subspace}_{cond}_N{n_mmlu}.pt
  outputs/<slug>_mmlu_baseline_N{n_mmlu}.pt
"""

import os
import sys
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats


# =========================================================================
# Helpers
# =========================================================================

def _alpha_tag(a: float) -> str:
    """360.0 -> '360';  3.5 -> '3_5'   (must match run_mitigation_sweep.sh)."""
    return str(int(a)) if float(a).is_integer() else str(a).replace(".", "_")


def _build_steered_name(slug: str, kind: str, layer_tag: str, alpha_tag: str,
                        scope: str, method: str, subspace: str,
                        cond_tag: str, maxex_tag: str) -> str:
    return (f"outputs/{slug}_{kind}_steered_"
            f"L{layer_tag}_a{alpha_tag}_{scope}_{method}_{subspace}_"
            f"{cond_tag}_N{maxex_tag}.pt")


def _build_mmlu_steered_name(slug: str, layer_tag: str, alpha_tag: str,
                             scope: str, method: str, subspace: str,
                             cond_tag: str, n_mmlu: int) -> str:
    return (f"outputs/{slug}_mmlu_"
            f"L{layer_tag}_a{alpha_tag}_{scope}_{method}_{subspace}_"
            f"{cond_tag}_N{n_mmlu}.pt")


def _safe_load(path: str):
    return torch.load(path, weights_only=False) if path and os.path.exists(path) else None


def _ex_id(r: dict):
    return r.get("id", r.get("example_id"))


# =========================================================================
# Outcome extractors  (1 = bias-positive, 0 = bias-clean, None = skip)
# =========================================================================

def _outcome_stereo(r: dict):
    v = r.get("prefers_stereotype")
    return int(bool(v)) if v is not None else None


def _outcome_bbq_ambig(r: dict):
    if r.get("condition") != "ambig":
        return None
    rt = r.get("response_type")
    if rt == "stereotyped_guess": return 1
    if rt == "correct":           return 0
    return None


# =========================================================================
# Paired rate computation -- KEY for subset mode
# =========================================================================

def paired_rates_and_mcnemar(
    baseline_results: List[dict],
    steered_results: List[dict],
    outcome_fn,
) -> Optional[Dict]:
    """
    Pair examples by id, compute baseline_rate / steered_rate / flip stats
    ONLY on the examples that appear in both files with valid outcomes.

    This is what makes subset mode correct: baseline is always the full
    dataset (say 2123 StereoSet examples), but the steered file may contain
    only a 2000-example subset. Computing the baseline rate naively from
    the full file would compare 2123 vs 2000 -- nonsense. Here we filter
    the baseline down to exactly the ids in the steered file.
    """
    b_map = {_ex_id(r): r for r in baseline_results}
    s_map = {_ex_id(r): r for r in steered_results}
    common_ids = sorted([k for k in b_map if k in s_map])

    yb, ys = [], []
    for k in common_ids:
        vb = outcome_fn(b_map[k])
        vs = outcome_fn(s_map[k])
        if vb is None or vs is None:
            continue
        yb.append(vb); ys.append(vs)

    if not yb:
        return None

    yb = np.array(yb); ys = np.array(ys)
    n = len(yb)
    unchanged = int(np.sum(yb == ys))
    b01 = int(np.sum((yb == 0) & (ys == 1)))   # baseline clean, steered biased (worse)
    b10 = int(np.sum((yb == 1) & (ys == 0)))   # baseline biased, steered clean (better)

    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        try:
            p = float(stats.binomtest(k, n=n_disc, p=0.5, alternative="two-sided").pvalue)
        except AttributeError:
            p = float(stats.binom_test(k, n=n_disc, p=0.5, alternative="two-sided"))

    return {
        "n_paired": n,
        "n_unchanged": unchanged,
        "n_flipped": n - unchanged,
        "b10_better": b10,
        "b01_worse": b01,
        "baseline_rate": float(yb.mean()),
        "steered_rate":  float(ys.mean()),
        "delta":         float(ys.mean() - yb.mean()),
        "mcnemar_p":     p,
    }


def verdict_for_dataset(r: Optional[Dict]) -> str:
    if r is None:                                             return "(no data)"
    if r["n_flipped"] == 0:                                   return "no effect (α too small?)"
    if r["n_flipped"] < 10:                                   return "barely any flips"
    if r["mcnemar_p"] < 0.05 and r["b10_better"] > r["b01_worse"]:
        ratio = r["b10_better"] / max(1, r["b01_worse"])
        if ratio >= 2.0: return f"STRONG directional ({ratio:.1f}×)"
        return f"directional ({ratio:.1f}×)"
    if r["mcnemar_p"] < 0.05 and r["b01_worse"] > r["b10_better"]:
        return "WORSE — steering increased bias"
    return "symmetric flips — direction not debiasing"


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--slug",  type=str, required=True)
    ap.add_argument("--scope", type=str, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--subspace", type=str, required=True)
    ap.add_argument("--layer", type=str, required=True)       # "25" or "20,24,25"
    ap.add_argument("--alphas", type=str, required=True)       # "200 360 500"
    ap.add_argument("--conditional", type=str, default="0")
    ap.add_argument("--max_examples_tag", type=str, default="full")
    ap.add_argument("--n_mmlu", type=int, default=500)
    ap.add_argument("--ss_base", type=str, required=True)
    ap.add_argument("--ga_base", type=str, required=True)
    ap.add_argument("--bbq_base", type=str, required=True)
    ap.add_argument("--mmlu_base", type=str, required=True)
    ap.add_argument("--emotion_vectors", type=str, required=True)
    ap.add_argument("--mitigation_directions", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    slug = args.slug
    layer_tag = args.layer.replace(",", "-")
    cond_tag = "cnd" if str(args.conditional) == "1" else "unc"
    alphas = [float(a) for a in args.alphas.split()]

    # Parse primary steering layer (first one, for direction lookup)
    primary_layer = int(args.layer.split(",")[0])

    # ---- Load baselines once ----
    ss_base_data = _safe_load(args.ss_base)
    ga_base_data = _safe_load(args.ga_base)
    bbq_base_data = _safe_load(args.bbq_base)
    mmlu_base_data = _safe_load(args.mmlu_base)

    # ---- Direction magnitudes at primary layer ----
    mit = _safe_load(args.mitigation_directions)
    dir_info = None
    if mit is not None:
        try:
            entry = mit["scopes"][args.scope][primary_layer]["methods"][args.method]
            sub_key = f"{args.subspace}_subspace"
            dir_info = {
                "norm_bias": entry[sub_key]["norm_bias"],
                "norm_perp": entry[sub_key]["norm_perp"],
                "overlap":   entry[sub_key]["overlap_ratio"],
                "norm_w":    float(np.linalg.norm(entry["w"])),
                "tau_suggest": entry["tau_suggest"],
            }
        except Exception as e:
            dir_info = {"error": str(e)}

    # ---- Hidden-state scale at steering layer from GoEmotions cache ----
    hnorm_info = None
    hidden_cache_path = f"outputs/{slug}_goemotions_hidden.pt"
    if os.path.exists(hidden_cache_path):
        try:
            hc = torch.load(hidden_cache_path, weights_only=False)
            if primary_layer in hc["hidden_states"]:
                H = hc["hidden_states"][primary_layer].float()
                norms = H.norm(dim=-1)
                hnorm_info = {
                    "mean": float(norms.mean()),
                    "median": float(norms.median()),
                    "p10": float(norms.quantile(0.10)),
                    "p90": float(norms.quantile(0.90)),
                    "n_examples": int(len(H)),
                }
        except Exception:
            pass

    # ---- Print header ----
    print()
    print("=" * 82)
    print("EMOTION-GUIDED BIAS MITIGATION — RESULTS SUMMARY")
    print("=" * 82)
    print("\nCONFIGURATION")
    print(f"  Model              : {args.model}")
    print(f"  Scope              : {args.scope}")
    print(f"  Method             : {args.method}")
    print(f"  Subspace           : {args.subspace}")
    print(f"  Conditional        : {bool(int(args.conditional))}")
    print(f"  Steering layer(s)  : {args.layer}")
    print(f"  Alphas tested      : {args.alphas}")
    print(f"  Max examples (bias): {args.max_examples_tag}")
    print(f"  MMLU questions     : {args.n_mmlu}")

    if dir_info is not None and "error" not in dir_info:
        print(f"\nDIRECTION MAGNITUDES  (scope={args.scope}, method={args.method}, "
              f"subspace={args.subspace}, layer={primary_layer})")
        print(f"  ||d_bias||         : {dir_info['norm_bias']:.4f}")
        print(f"  ||d_perp||         : {dir_info['norm_perp']:.4f}")
        print(f"  Overlap ratio      : {dir_info['overlap']:.4f}   "
              f"({100*dir_info['overlap']:.1f}% of d_bias inside protected subspace)")
        print(f"  ||w|| (feat-space) : {dir_info['norm_w']:.4f}")
        print(f"  tau_suggest        : {dir_info['tau_suggest']:.4f}")
    elif dir_info is not None:
        print(f"\n[warn] Could not read directions at layer {primary_layer}: {dir_info['error']}")

    if hnorm_info is not None:
        print(f"\nHIDDEN-STATE SCALE AT LAYER {primary_layer}  "
              f"(from GoEmotions cache, N={hnorm_info['n_examples']})")
        print(f"  mean ||h||   : {hnorm_info['mean']:.1f}")
        print(f"  median ||h|| : {hnorm_info['median']:.1f}")
        print(f"  p10 / p90    : {hnorm_info['p10']:.1f} / {hnorm_info['p90']:.1f}")
    else:
        print(f"\n[info] Hidden-state scale unavailable (no GoEmotions cache "
              f"at {hidden_cache_path}).")

    # ---- Per-alpha rows ----
    rows = []

    print()
    print("-" * 82)
    print("BIAS PROBE + MMLU — per alpha, paired by example id")
    print("-" * 82)

    for alpha in alphas:
        atag = _alpha_tag(alpha)
        ss_path  = _build_steered_name(slug, "stereoset", layer_tag, atag,
                                       args.scope, args.method, args.subspace,
                                       cond_tag, args.max_examples_tag)
        ga_path  = _build_steered_name(slug, "genassoc", layer_tag, atag,
                                       args.scope, args.method, args.subspace,
                                       cond_tag, args.max_examples_tag)
        bbq_path = _build_steered_name(slug, "bbq", layer_tag, atag,
                                       args.scope, args.method, args.subspace,
                                       cond_tag, args.max_examples_tag)
        mmlu_path = _build_mmlu_steered_name(slug, layer_tag, atag,
                                             args.scope, args.method, args.subspace,
                                             cond_tag, args.n_mmlu)

        ss_steer  = _safe_load(ss_path)
        ga_steer  = _safe_load(ga_path)
        bbq_steer = _safe_load(bbq_path)
        mmlu_steer = _safe_load(mmlu_path)

        # Effective perturbation
        if dir_info is not None and "error" not in dir_info:
            perturb = alpha * dir_info["norm_perp"]
            perturb_pct = (100 * perturb / hnorm_info["mean"]) if hnorm_info else None
        else:
            perturb, perturb_pct = None, None

        # Bias stats
        ss_stat = paired_rates_and_mcnemar(
            ss_base_data["results"] if ss_base_data else [],
            ss_steer["results"] if ss_steer else [],
            _outcome_stereo,
        ) if ss_steer is not None else None

        ga_stat = paired_rates_and_mcnemar(
            ga_base_data["results"] if ga_base_data else [],
            ga_steer["results"] if ga_steer else [],
            _outcome_stereo,
        ) if ga_steer is not None else None

        bbq_stat = paired_rates_and_mcnemar(
            bbq_base_data["results"] if bbq_base_data else [],
            bbq_steer["results"] if bbq_steer else [],
            _outcome_bbq_ambig,
        ) if bbq_steer is not None else None

        # MMLU
        mmlu_base_acc = mmlu_base_data["accuracy"] if mmlu_base_data else None
        mmlu_steer_acc = mmlu_steer["accuracy"] if mmlu_steer else None
        mmlu_delta = ((mmlu_steer_acc - mmlu_base_acc)
                      if (mmlu_base_acc is not None and mmlu_steer_acc is not None) else None)
        mmlu_pct_loss = ((100 * (-mmlu_delta) / mmlu_base_acc)
                         if (mmlu_delta is not None and mmlu_base_acc and mmlu_delta < 0) else 0.0)

        # ---- Print block per alpha ----
        print(f"\n==  alpha = {alpha}   layer = {args.layer}  ==")
        if perturb is not None:
            pct_str = f"{perturb_pct:.1f}% of ||h||" if perturb_pct is not None else "(||h|| unknown)"
            print(f"   α × ||d_perp|| = {perturb:.2f}   ({pct_str})")

        def _row(name, st):
            if st is None:
                return f"   {name:<12}  (no paired data)"
            return (f"   {name:<12}  "
                    f"base={st['baseline_rate']:.4f}  "
                    f"steer={st['steered_rate']:.4f}  "
                    f"Δ={st['delta']:+.4f}  "
                    f"N={st['n_paired']}  "
                    f"b10/b01={st['b10_better']}/{st['b01_worse']}  "
                    f"p={st['mcnemar_p']:.3g}  "
                    f"-- {verdict_for_dataset(st)}")

        print(_row("StereoSet",  ss_stat))
        print(_row("GenAssoc",   ga_stat))
        print(_row("BBQ-ambig",  bbq_stat))

        if mmlu_base_acc is not None and mmlu_steer_acc is not None:
            print(f"   {'MMLU':<12}  "
                  f"base={mmlu_base_acc:.4f}  "
                  f"steer={mmlu_steer_acc:.4f}  "
                  f"Δ={mmlu_delta:+.4f}  "
                  f"loss={mmlu_pct_loss:.2f}%  "
                  f"N={args.n_mmlu}")
        else:
            print(f"   {'MMLU':<12}  (missing baseline or steered file)")

        # Per-alpha overall verdict
        reductions = []
        for st in (ss_stat, ga_stat, bbq_stat):
            if st and st["delta"] < 0:
                reductions.append(-st["delta"])
        max_red = max(reductions) * 100 if reductions else 0.0
        n_reduced = sum(1 for st in (ss_stat, ga_stat, bbq_stat)
                        if st and st["delta"] < 0)
        sig_reduced = sum(1 for st in (ss_stat, ga_stat, bbq_stat)
                          if st and st["delta"] < 0 and st["mcnemar_p"] < 0.05)

        if mmlu_pct_loss > 10:
            tradeoff = "CAPABILITY DAMAGED — discard"
        elif sig_reduced >= 2 and mmlu_pct_loss < 2:
            tradeoff = "STRONG CANDIDATE (significant reduction on ≥2 datasets, <2% MMLU loss)"
        elif sig_reduced >= 1 and mmlu_pct_loss < 5:
            tradeoff = "promising"
        elif max_red < 0.5:
            tradeoff = "too weak — try larger alpha or different layer"
        else:
            tradeoff = "mixed — some reduction but not significant OR some capability cost"

        print(f"   VERDICT      : {tradeoff}")

        rows.append({
            "alpha": alpha,
            "layer": args.layer,
            "effective_perturb": perturb,
            "perturb_pct_of_h":  perturb_pct,
            "ss_base_rate":  ss_stat["baseline_rate"] if ss_stat else None,
            "ss_steer_rate": ss_stat["steered_rate"]  if ss_stat else None,
            "ss_delta":      ss_stat["delta"]         if ss_stat else None,
            "ss_b10":        ss_stat["b10_better"]    if ss_stat else None,
            "ss_b01":        ss_stat["b01_worse"]     if ss_stat else None,
            "ss_p":          ss_stat["mcnemar_p"]     if ss_stat else None,
            "ga_base_rate":  ga_stat["baseline_rate"] if ga_stat else None,
            "ga_steer_rate": ga_stat["steered_rate"]  if ga_stat else None,
            "ga_delta":      ga_stat["delta"]         if ga_stat else None,
            "ga_b10":        ga_stat["b10_better"]    if ga_stat else None,
            "ga_b01":        ga_stat["b01_worse"]     if ga_stat else None,
            "ga_p":          ga_stat["mcnemar_p"]     if ga_stat else None,
            "bbq_base_rate": bbq_stat["baseline_rate"] if bbq_stat else None,
            "bbq_steer_rate": bbq_stat["steered_rate"] if bbq_stat else None,
            "bbq_delta":     bbq_stat["delta"]        if bbq_stat else None,
            "bbq_b10":       bbq_stat["b10_better"]   if bbq_stat else None,
            "bbq_b01":       bbq_stat["b01_worse"]    if bbq_stat else None,
            "bbq_p":         bbq_stat["mcnemar_p"]    if bbq_stat else None,
            "mmlu_base":  mmlu_base_acc,
            "mmlu_steer": mmlu_steer_acc,
            "mmlu_delta": mmlu_delta,
            "mmlu_pct_loss": mmlu_pct_loss,
            "verdict": tradeoff,
        })

    # ---- Compact leaderboard ----
    print()
    print("-" * 82)
    print("LEADERBOARD")
    print("-" * 82)
    print(f"{'α':>6} | {'SS Δ':>8} {'GA Δ':>8} {'BBQ Δ':>8} | {'MMLU Δ':>8} {'loss%':>7} | verdict")
    print("-" * 82)

    def fmt(v, w=8, d=4):
        return (" " * (w - 3) + "N/A") if v is None else f"{v:>+{w}.{d}f}"

    def fmt_pct(v):
        return "    N/A" if v is None else f"{v:>6.2f}%"

    for row in rows:
        print(f"{row['alpha']:>6.1f} | "
              f"{fmt(row['ss_delta'])} {fmt(row['ga_delta'])} {fmt(row['bbq_delta'])} | "
              f"{fmt(row['mmlu_delta'])} {fmt_pct(row['mmlu_pct_loss'])} | "
              f"{row['verdict']}")
    print("=" * 82)

    # ---- Save CSV ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
