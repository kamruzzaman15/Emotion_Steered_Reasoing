"""
analyze_mitigation.py -- Pre/post comparison of mitigation results.

Compares baseline probe results (produced by your existing probe_*.py scripts)
against steered probe results (from probe_with_steering.py). Computes:

  1. Bias rate (primary outcome):
       - StereoSet / GenAssocBias: % preferring stereotype
       - BBQ-ambig: % stereotyped_guess (among correct + stereotyped_guess)
  2. Δ emotion activation profile at the analysis layer: steered − baseline,
     separated by key response subgroups.
  3. Statistical test on the change in bias rate (McNemar's test when we can
     match examples by id; chi-square otherwise).
  4. Side-by-side figures saved to the output directory.

Intentionally DOES NOT modify analyze_results.py. This is an additive module.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ============================================================
# Metric extractors
# ============================================================

def bias_rate_stereoset_style(results: List[dict]) -> Tuple[float, int, int]:
    """Fraction preferring stereotype. Returns (rate, n_stereo, n_total)."""
    n = len(results)
    if n == 0:
        return float("nan"), 0, 0
    n_s = sum(1 for r in results if r.get("prefers_stereotype"))
    return n_s / n, n_s, n


def bias_rate_bbq_ambig(results: List[dict]) -> Tuple[float, int, int]:
    """Fraction of stereotyped_guess among correct + stereotyped_guess on BBQ-ambig."""
    amb = [r for r in results if r.get("condition") == "ambig"]
    n_sg = sum(1 for r in amb if r.get("response_type") == "stereotyped_guess")
    n_co = sum(1 for r in amb if r.get("response_type") == "correct")
    denom = n_sg + n_co
    if denom == 0:
        return float("nan"), 0, 0
    return n_sg / denom, n_sg, denom


def mean_emotion_activations_stereoset_style(
    results: List[dict],
    emotions: List[str],
    layer: int,
    condition: str = "stereotype",
) -> Dict[str, float]:
    """Mean projection per emotion at the given layer, for the given condition."""
    acc = {e: [] for e in emotions}
    for r in results:
        proj = r.get("emotion_projections", {}).get(condition, {}).get(layer, {})
        for e in emotions:
            if e in proj:
                acc[e].append(proj[e])
    return {e: float(np.mean(v)) if v else float("nan") for e, v in acc.items()}


def mean_emotion_activations_bbq(
    results: List[dict],
    emotions: List[str],
    layer: int,
    response_type: Optional[str] = None,
) -> Dict[str, float]:
    """Mean projection per emotion at the given layer, optionally filtered by response_type."""
    subset = [r for r in results if response_type is None or r.get("response_type") == response_type]
    acc = {e: [] for e in emotions}
    for r in subset:
        proj = r.get("emotion_projections_at_question", {}).get(layer, {})
        for e in emotions:
            if e in proj:
                acc[e].append(proj[e])
    return {e: float(np.mean(v)) if v else float("nan") for e, v in acc.items()}


# ============================================================
# Matching for McNemar
# ============================================================

def paired_outcomes_by_id(
    base_results: List[dict],
    steer_results: List[dict],
    outcome_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pair examples by id; return matched outcome arrays (0/1) of equal length.
    `outcome_fn(result) -> int in {0,1} | None` (None examples skipped).
    """
    b_map = {r.get("id", r.get("example_id")): r for r in base_results}
    s_map = {r.get("id", r.get("example_id")): r for r in steer_results}
    common = sorted(set(b_map.keys()) & set(s_map.keys()))
    b_vals, s_vals = [], []
    for k in common:
        bv = outcome_fn(b_map[k])
        sv = outcome_fn(s_map[k])
        if bv is None or sv is None:
            continue
        b_vals.append(bv)
        s_vals.append(sv)
    return np.array(b_vals), np.array(s_vals)


def mcnemar_change(base_y: np.ndarray, steer_y: np.ndarray) -> Dict[str, float]:
    """
    McNemar's test on matched binary outcomes (1=bias-positive).
    Returns dict with counts and p-value.
    """
    if len(base_y) == 0:
        return {"n": 0, "b01": 0, "b10": 0, "p_value": float("nan"),
                "baseline_rate": float("nan"), "steered_rate": float("nan"),
                "delta": float("nan")}
    # b01: baseline=0, steered=1  (got worse under steering)
    # b10: baseline=1, steered=0  (got better under steering)
    b01 = int(np.sum((base_y == 0) & (steer_y == 1)))
    b10 = int(np.sum((base_y == 1) & (steer_y == 0)))
    # Exact binomial test on b10 among discordant pairs
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        # Exact binomial p-value (two-sided)
        k = min(b01, b10)
        # Python 3.8+: stats.binom_test deprecated in newer SciPy. Use binomtest.
        try:
            res = stats.binomtest(k, n=n_disc, p=0.5, alternative="two-sided")
            p = float(res.pvalue)
        except AttributeError:
            p = float(stats.binom_test(k, n=n_disc, p=0.5, alternative="two-sided"))
    return {
        "n": int(len(base_y)),
        "b01_worse": b01,
        "b10_better": b10,
        "p_value": float(p),
        "baseline_rate": float(base_y.mean()),
        "steered_rate": float(steer_y.mean()),
        "delta": float(steer_y.mean() - base_y.mean()),
    }


# ============================================================
# Figures
# ============================================================

def plot_rate_comparison(
    labels: List[str], baseline: List[float], steered: List[float], title: str, out_path: str,
):
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, np.array(baseline) * 100, width=w, color="#9e9e9e", label="baseline")
    ax.bar(x + w / 2, np.array(steered) * 100, width=w, color="#d32f2f", label="steered")
    for i, (b, s) in enumerate(zip(baseline, steered)):
        ax.text(i - w / 2, b * 100 + 0.5, f"{100*b:.1f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, s * 100 + 0.5, f"{100*s:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bias rate (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_emotion_shift(
    emotions: List[str],
    baseline: Dict[str, float],
    steered: Dict[str, float],
    title: str,
    out_path: str,
):
    diffs = pd.Series(
        {e: steered[e] - baseline[e] for e in emotions if not np.isnan(baseline[e]) and not np.isnan(steered[e])}
    ).sort_values()
    if diffs.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(diffs))))
    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in diffs.values]
    diffs.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean activation: steered − baseline")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# Per-dataset analysis
# ============================================================

def analyze_stereoset_style(
    base_data: dict,
    steer_data: dict,
    out_dir: str,
    tag: str,
    model_short: str,
):
    """Works for StereoSet and GenAssocBias (same result schema)."""
    base_res = base_data["results"]
    steer_res = steer_data["results"]
    emotions = base_data["emotions"]
    layers = base_data["target_layers"]
    analysis_layer = layers[len(layers) * 2 // 3]

    # Overall bias rate
    br_b, ns_b, n_b = bias_rate_stereoset_style(base_res)
    br_s, ns_s, n_s = bias_rate_stereoset_style(steer_res)
    print(f"\n[{tag}] Bias rate:   baseline={br_b:.3f} ({ns_b}/{n_b})   "
          f"steered={br_s:.3f} ({ns_s}/{n_s})   Δ={br_s - br_b:+.3f}")

    # McNemar (paired by id)
    def _outcome(r):
        v = r.get("prefers_stereotype")
        return int(bool(v)) if v is not None else None
    yb, ys = paired_outcomes_by_id(base_res, steer_res, _outcome)
    stat = mcnemar_change(yb, ys)
    print(f"[{tag}] McNemar paired: N={stat['n']}  b01={stat['b01_worse']}  "
          f"b10={stat['b10_better']}  p={stat['p_value']:.4g}  "
          f"Δrate={stat['delta']:+.4f}")

    # Per-bias-type comparison
    type_rows = []
    for r_b, r_s_list in [("baseline", base_res), ("steered", steer_res)]:
        pass
    types_present = sorted({r.get("bias_type", "unknown") for r in base_res})
    per_type_base, per_type_steer, labels_keep = [], [], []
    for bt in types_present:
        rb = [r for r in base_res if r.get("bias_type", "unknown") == bt]
        rs = [r for r in steer_res if r.get("bias_type", "unknown") == bt]
        if len(rb) < 10 or len(rs) < 10:
            continue
        bb, _, _ = bias_rate_stereoset_style(rb)
        bs, _, _ = bias_rate_stereoset_style(rs)
        per_type_base.append(bb); per_type_steer.append(bs); labels_keep.append(bt)
        type_rows.append({
            "model": model_short, "dataset": tag, "bias_type": bt,
            "baseline_rate": bb, "steered_rate": bs, "delta": bs - bb,
            "n_base": len(rb), "n_steer": len(rs),
        })

    if labels_keep:
        # Overall + per-type bar chart
        labels_full = ["ALL"] + labels_keep
        base_full = [br_b] + per_type_base
        steer_full = [br_s] + per_type_steer
        plot_rate_comparison(
            labels_full, base_full, steer_full,
            title=f"[{model_short}] {tag}: % prefers_stereotype (baseline vs steered)",
            out_path=os.path.join(out_dir, f"{model_short}_mit_{tag}_bias_rate.png"),
        )

    # Emotion shift at analysis_layer, on the "stereotype" condition
    base_em = mean_emotion_activations_stereoset_style(base_res, emotions, analysis_layer, "stereotype")
    steer_em = mean_emotion_activations_stereoset_style(steer_res, emotions, analysis_layer, "stereotype")
    plot_emotion_shift(
        emotions, base_em, steer_em,
        title=f"[{model_short}] {tag}: emotion Δ (steered − baseline) @ layer {analysis_layer}",
        out_path=os.path.join(out_dir, f"{model_short}_mit_{tag}_emotion_shift_L{analysis_layer}.png"),
    )

    # CSV summary
    summary = {
        "model": model_short, "dataset": tag,
        "baseline_rate": br_b, "steered_rate": br_s,
        "delta_rate": br_s - br_b,
        "mcnemar_n": stat["n"],
        "mcnemar_b01_worse": stat["b01_worse"],
        "mcnemar_b10_better": stat["b10_better"],
        "mcnemar_p_value": stat["p_value"],
        "analysis_layer": analysis_layer,
        **steer_data.get("steering", {}),
    }
    rows = [summary]
    if type_rows:
        rows.extend(type_rows)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"{model_short}_mit_{tag}_summary.csv"), index=False
    )

    return summary, type_rows


def analyze_bbq(
    base_data: dict,
    steer_data: dict,
    out_dir: str,
    model_short: str,
):
    base_res = base_data["results"]
    steer_res = steer_data["results"]
    emotions = base_data["emotions"]
    layers = base_data["target_layers"]
    analysis_layer = layers[len(layers) * 2 // 3]

    br_b, ns_b, n_b = bias_rate_bbq_ambig(base_res)
    br_s, ns_s, n_s = bias_rate_bbq_ambig(steer_res)
    print(f"\n[bbq_ambig] Stereotyped-guess rate (vs correct): "
          f"baseline={br_b:.3f} ({ns_b}/{n_b})   steered={br_s:.3f} ({ns_s}/{n_s})   "
          f"Δ={br_s - br_b:+.3f}")

    # Paired outcome: among examples where both baseline and steered produced
    # one of {correct, stereotyped_guess} in the ambig condition.
    def _outcome(r):
        if r.get("condition") != "ambig":
            return None
        rt = r.get("response_type")
        if rt == "stereotyped_guess":
            return 1
        if rt == "correct":
            return 0
        return None

    yb, ys = paired_outcomes_by_id(base_res, steer_res, _outcome)
    stat = mcnemar_change(yb, ys)
    print(f"[bbq_ambig] McNemar paired: N={stat['n']}  b01={stat['b01_worse']}  "
          f"b10={stat['b10_better']}  p={stat['p_value']:.4g}  "
          f"Δrate={stat['delta']:+.4f}")

    # Per-category comparison
    cats = sorted({r.get("category", "unknown") for r in base_res if r.get("condition") == "ambig"})
    per_cat_base, per_cat_steer, labels_keep, rows = [], [], [], []
    for c in cats:
        rb = [r for r in base_res if r.get("category") == c]
        rs = [r for r in steer_res if r.get("category") == c]
        bb, _, nbb = bias_rate_bbq_ambig(rb)
        bs, _, nss = bias_rate_bbq_ambig(rs)
        if nbb < 10 or nss < 10:
            continue
        per_cat_base.append(bb); per_cat_steer.append(bs); labels_keep.append(c)
        rows.append({
            "model": model_short, "dataset": "bbq_ambig", "category": c,
            "baseline_rate": bb, "steered_rate": bs, "delta": bs - bb,
            "n_base": nbb, "n_steer": nss,
        })

    if labels_keep:
        plot_rate_comparison(
            ["ALL"] + labels_keep,
            [br_b] + per_cat_base,
            [br_s] + per_cat_steer,
            title=f"[{model_short}] BBQ-ambig: stereotyped_guess / (stereotyped_guess+correct)",
            out_path=os.path.join(out_dir, f"{model_short}_mit_bbq_ambig_bias_rate.png"),
        )

    # Emotion shift on stereotyped_guess subset at analysis layer
    base_em = mean_emotion_activations_bbq(base_res, emotions, analysis_layer, "stereotyped_guess")
    steer_em = mean_emotion_activations_bbq(steer_res, emotions, analysis_layer, "stereotyped_guess")
    plot_emotion_shift(
        emotions, base_em, steer_em,
        title=f"[{model_short}] BBQ-ambig (stereotyped_guess): emotion Δ @ layer {analysis_layer}",
        out_path=os.path.join(out_dir, f"{model_short}_mit_bbq_ambig_emotion_shift_L{analysis_layer}.png"),
    )

    summary = {
        "model": model_short, "dataset": "bbq_ambig",
        "baseline_rate": br_b, "steered_rate": br_s,
        "delta_rate": br_s - br_b,
        "mcnemar_n": stat["n"],
        "mcnemar_b01_worse": stat["b01_worse"],
        "mcnemar_b10_better": stat["b10_better"],
        "mcnemar_p_value": stat["p_value"],
        "analysis_layer": analysis_layer,
        **steer_data.get("steering", {}),
    }
    all_rows = [summary] + rows
    pd.DataFrame(all_rows).to_csv(
        os.path.join(out_dir, f"{model_short}_mit_bbq_ambig_summary.csv"), index=False
    )
    return summary, rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--stereoset_baseline", type=str, default=None)
    parser.add_argument("--stereoset_steered",  type=str, default=None)
    parser.add_argument("--genassoc_baseline",  type=str, default=None)
    parser.add_argument("--genassoc_steered",   type=str, default=None)
    parser.add_argument("--bbq_baseline",       type=str, default=None)
    parser.add_argument("--bbq_steered",        type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = []

    def _load(p):
        return torch.load(p, weights_only=False) if p and os.path.exists(p) else None

    pairs = [
        ("stereoset", args.stereoset_baseline, args.stereoset_steered),
        ("genassoc",  args.genassoc_baseline,  args.genassoc_steered),
    ]
    for tag, bp, sp in pairs:
        base_data = _load(bp); steer_data = _load(sp)
        if base_data is None or steer_data is None:
            if bp or sp:
                print(f"[{tag}] skipped (missing baseline or steered path)")
            continue
        summ, type_rows = analyze_stereoset_style(
            base_data, steer_data, args.output_dir, tag, args.model_short
        )
        all_summaries.append(summ)

    base_bbq = _load(args.bbq_baseline); steer_bbq = _load(args.bbq_steered)
    if base_bbq is not None and steer_bbq is not None:
        summ, _ = analyze_bbq(base_bbq, steer_bbq, args.output_dir, args.model_short)
        all_summaries.append(summ)
    elif args.bbq_baseline or args.bbq_steered:
        print("[bbq_ambig] skipped (missing baseline or steered path)")

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(args.output_dir, f"{args.model_short}_mit_ALL_summary.csv"), index=False
        )
        print(f"\n[analyze_mitigation] Wrote overall summary to "
              f"{args.output_dir}/{args.model_short}_mit_ALL_summary.csv")


if __name__ == "__main__":
    main()
