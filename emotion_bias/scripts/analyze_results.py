# """
# analyze_results.py - Statistical analysis and figure generation.

# Filename convention:
#     {model_short}_{dataset}_{figtype}_{description}.pdf
# e.g. gemma2_2b_stereoset_fig2_emotion_diff.pdf
#      gemma2_2b_bbq_ambig_fig5_response_distribution.pdf
#      gemma2_2b_genassocbias_fig9_predictive.pdf
#      gemma2_2b_cross_fig10_comparison.pdf
# """

# import os
# import re
# import json
# import argparse
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score


# # ============================================================
# # Helpers
# # ============================================================

# def load_results(path):
#     return torch.load(path, weights_only=False)


# # Global stats accumulator across all analyses
# ALL_STATS = []


# def add_stats(rows, name):
#     for r in rows:
#         r["section"] = name
#         ALL_STATS.append(r)


# def shorten_slug(slug: str) -> str:
#     """Convert HF-style slug into a short filename-friendly identifier.

#     Examples:
#       google_gemma_2_2b_it                  -> gemma2_2b
#       meta_llama_llama_3.2_3b_instruct      -> llama32_3b
#       mistralai_mistral_7b_instruct_v0.3    -> mistral_7b
#       qwen_qwen2.5_7b_instruct              -> qwen25_7b
#       meta_llama_meta_llama_3.1_8b_instruct -> llama31_8b
#     """
#     s = slug.lower()
#     prefixes = ["google_", "meta_llama_", "mistralai_", "qwen_",
#                 "anthropic_", "microsoft_", "openai_"]
#     suffixes = ["_it", "_instruct", "_chat", "_hf", "_base"]
#     version_re = re.compile(r"_v\d+(?:[._]\d+)*$")

#     # Strip ORG prefix exactly once (don't re-strip — model name may legitimately
#     # start with the same string, e.g. "meta_llama_meta_llama_3.1_8b_instruct").
#     for p in prefixes:
#         if s.startswith(p):
#             s = s[len(p):]
#             break

#     # Strip suffixes + version markers iteratively (order between them matters).
#     changed = True
#     while changed:
#         changed = False
#         for sfx in suffixes:
#             if s.endswith(sfx):
#                 s = s[:-len(sfx)]; changed = True
#         new = version_re.sub("", s)
#         if new != s:
#             s = new; changed = True

#     # Compact version numbers and normalize "meta_llama" -> "llama":
#     #   gemma_2_2b -> gemma2_2b
#     #   llama_3.2_3b -> llama32_3b
#     #   meta_llama_3.1_8b -> llama31_8b
#     #   qwen2.5_7b -> qwen25_7b
#     s = re.sub(
#         r"^(?:meta_)?(gemma|llama|qwen|mistral|phi|mixtral)_?(\d+(?:\.\d+)?)_",
#         lambda m: f"{m.group(1)}{m.group(2).replace('.', '')}_",
#         s,
#     )
#     return s


# def fig_path(output_dir, model_short, dataset, figtype, description, ext="pdf"):
#     """Build a figure path with the new convention."""
#     fname = f"{model_short}_{dataset}_{figtype}_{description}.{ext}"
#     return os.path.join(output_dir, fname)


# def save_fig(output_dir, model_short, dataset, figtype, description):
#     """Save current matplotlib figure as both PDF and PNG."""
#     plt.savefig(fig_path(output_dir, model_short, dataset, figtype, description, "pdf"), dpi=150)
#     plt.savefig(fig_path(output_dir, model_short, dataset, figtype, description, "png"), dpi=150)
#     plt.close()


# def save_stats_csv(rows, output_dir, model_short, name):
#     if not rows:
#         return
#     df = pd.DataFrame(rows)
#     path = os.path.join(output_dir, f"{model_short}_stats_{name}.csv")
#     df.to_csv(path, index=False)
#     print(f"  Stats saved: {path}")
#     add_stats(rows, name)


# def _sig(p):
#     return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# # ============================================================
# # Generic 3-condition analysis (used by StereoSet AND GenAssocBias)
# # ============================================================

# def _three_condition_analysis(
#     data, output_dir, model_short, dataset_name,
#     fig_prefix_letter="",
# ):
#     """
#     Produce the standard 5-figure suite for any dataset that has
#     {stereotype, anti_stereotype, unrelated} conditions:
#       fig1  - Mean activation by condition (grouped bars)
#       fig2  - stereotype - anti_stereotype difference (horizontal bars)
#       fig3  - per-bias-type difference (2x2 grid)
#       fig4  - layer-wise top-5 emotions
#       fig9  - predictive importance (LR coefficients)
#     """
#     results = data["results"]
#     emotions = data["emotions"]
#     layers = data["target_layers"]
#     analysis_layer = layers[len(layers) * 2 // 3]

#     print(f"\n[{model_short}] ANALYSIS: {dataset_name} (layer {analysis_layer})")

#     rows = []
#     for r in results:
#         for condition in ["stereotype", "anti_stereotype", "unrelated"]:
#             proj = r["emotion_projections"][condition].get(analysis_layer, {})
#             for emotion, value in proj.items():
#                 rows.append({
#                     "id": r.get("id"),
#                     "bias_type": r.get("bias_type", "unknown"),
#                     "condition": condition,
#                     "prefers_stereotype": r.get("prefers_stereotype"),
#                     "emotion": emotion,
#                     "activation": value,
#                 })
#     df = pd.DataFrame(rows)
#     if df.empty:
#         print(f"  [skip] empty dataframe for {dataset_name}")
#         return

#     # ---- fig1: by condition ----
#     fig, ax = plt.subplots(figsize=(14, 6))
#     pivot = df.groupby(["emotion", "condition"])["activation"].mean().reset_index()
#     pivot_wide = pivot.pivot(index="emotion", columns="condition", values="activation")
#     pivot_wide = pivot_wide.reindex(columns=["stereotype", "anti_stereotype", "unrelated"])
#     pivot_wide.plot(kind="bar", ax=ax, width=0.8)
#     ax.set_title(f"[{model_short}] {dataset_name}: Mean Emotion Activation by Condition (Layer {analysis_layer})")
#     ax.set_ylabel("Mean Projection onto Emotion Vector")
#     ax.set_xlabel("")
#     ax.legend(title="Condition")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     save_fig(output_dir, model_short, dataset_name, f"fig1{fig_prefix_letter}", "emotion_by_condition")

#     # ---- fig2: stereo - anti diff ----
#     s_means = df[df["condition"] == "stereotype"].groupby("emotion")["activation"].mean()
#     a_means = df[df["condition"] == "anti_stereotype"].groupby("emotion")["activation"].mean()
#     diff = (s_means - a_means).sort_values()

#     fig, ax = plt.subplots(figsize=(10, 10))
#     colors = ["#d32f2f" if v > 0 else "#1976d2" for v in diff.values]
#     diff.plot(kind="barh", ax=ax, color=colors)
#     ax.set_title(f"[{model_short}] {dataset_name}: Stereotype − Anti-stereotype")
#     ax.set_xlabel("Mean Activation Difference")
#     ax.axvline(x=0, color="black", linewidth=0.8)
#     plt.tight_layout()
#     save_fig(output_dir, model_short, dataset_name, f"fig2{fig_prefix_letter}", "emotion_diff")

#     # ---- fig3: per-bias-type 2x2 ----
#     bias_types = [b for b in df["bias_type"].unique() if pd.notna(b)]
#     n_types = min(4, len(bias_types))
#     if n_types > 0:
#         n_rows = max(1, int(np.ceil(n_types / 2)))
#         fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), squeeze=False)
#         for idx in range(n_types):
#             bt = bias_types[idx]
#             ax = axes[idx // 2][idx % 2]
#             subset = df[df["bias_type"] == bt]
#             sm = subset[subset["condition"] == "stereotype"].groupby("emotion")["activation"].mean()
#             am = subset[subset["condition"] == "anti_stereotype"].groupby("emotion")["activation"].mean()
#             diff_bt = (sm - am).sort_values()
#             colors = ["#d32f2f" if v > 0 else "#1976d2" for v in diff_bt.values]
#             diff_bt.plot(kind="barh", ax=ax, color=colors)
#             ax.set_title(f"Bias Type: {bt}", fontsize=12)
#             ax.axvline(x=0, color="black", linewidth=0.8)
#         # Hide unused axes
#         for idx in range(n_types, n_rows * 2):
#             axes[idx // 2][idx % 2].axis("off")
#         plt.suptitle(f"[{model_short}] {dataset_name}: Difference by Bias Type", fontsize=14)
#         plt.tight_layout()
#         save_fig(output_dir, model_short, dataset_name, f"fig3{fig_prefix_letter}", "by_bias_type")

#     # ---- t-tests ----
#     stat_rows = []
#     for emotion in sorted(emotions):
#         s_vals = df[(df["condition"] == "stereotype") & (df["emotion"] == emotion)]["activation"].values
#         a_vals = df[(df["condition"] == "anti_stereotype") & (df["emotion"] == emotion)]["activation"].values
#         if len(s_vals) > 0 and len(a_vals) > 0:
#             n = min(len(s_vals), len(a_vals))
#             t_stat, p_val = stats.ttest_rel(s_vals[:n], a_vals[:n])
#             stat_rows.append({
#                 "model": model_short, "dataset": dataset_name,
#                 "analysis": "stereotype_vs_anti_stereotype",
#                 "layer": analysis_layer, "emotion": emotion,
#                 "stereo_mean": float(np.mean(s_vals)),
#                 "anti_mean": float(np.mean(a_vals)),
#                 "diff": float(np.mean(s_vals) - np.mean(a_vals)),
#                 "t_stat": float(t_stat), "p_value": float(p_val),
#                 "significance": _sig(p_val),
#                 "n_stereo": int(len(s_vals)), "n_anti": int(len(a_vals)),
#             })
#     save_stats_csv(stat_rows, output_dir, model_short, f"{dataset_name}_ttests")

#     # ---- fig4: layer-wise top 5 ----
#     sig_sorted = sorted(
#         [(r["emotion"], r["diff"], r["p_value"]) for r in stat_rows if r["p_value"] < 0.05],
#         key=lambda x: abs(x[1]), reverse=True,
#     )
#     top_emotions = [e for e, _, _ in sig_sorted[:5]]
#     if not top_emotions:
#         # Fallback to top-5 by absolute diff regardless of significance
#         top_emotions = sorted(stat_rows, key=lambda r: abs(r["diff"]), reverse=True)[:5]
#         top_emotions = [r["emotion"] for r in top_emotions]
#     if top_emotions:
#         fig, ax = plt.subplots(figsize=(12, 6))
#         for emotion in top_emotions:
#             diffs_by_layer = []
#             for layer in layers:
#                 s_vals, a_vals = [], []
#                 for r in results:
#                     s_proj = r["emotion_projections"]["stereotype"].get(layer, {})
#                     a_proj = r["emotion_projections"]["anti_stereotype"].get(layer, {})
#                     if emotion in s_proj and emotion in a_proj:
#                         s_vals.append(s_proj[emotion])
#                         a_vals.append(a_proj[emotion])
#                 diffs_by_layer.append(np.mean(s_vals) - np.mean(a_vals) if s_vals else 0)
#             ax.plot(layers, diffs_by_layer, marker="o", label=emotion, linewidth=2)
#         ax.set_xlabel("Layer")
#         ax.set_ylabel("Activation Difference (Stereo − Anti)")
#         ax.set_title(f"[{model_short}] {dataset_name}: Layer-wise (Top 5 emotions)")
#         ax.legend()
#         ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
#         plt.tight_layout()
#         save_fig(output_dir, model_short, dataset_name, f"fig4{fig_prefix_letter}", "layerwise")

#     # ---- fig9: predictive importance ----
#     X_rows, y = [], []
#     for r in results:
#         proj = r["emotion_projections"]["stereotype"].get(analysis_layer, {})
#         if len(proj) == len(emotions):
#             X_rows.append([proj[e] for e in sorted(emotions)])
#             y.append(1 if r.get("prefers_stereotype") else 0)
#     X = np.array(X_rows)
#     y = np.array(y)

#     if len(X) >= 20 and len(set(y)) == 2:
#         clf = LogisticRegression(max_iter=1000, random_state=42)
#         cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
#         clf.fit(X, y)
#         coefs = clf.coef_[0]
#         importance = pd.Series(coefs, index=sorted(emotions)).sort_values()

#         fig, ax = plt.subplots(figsize=(10, 10))
#         colors = ["#d32f2f" if v > 0 else "#1976d2" for v in importance.values]
#         importance.plot(kind="barh", ax=ax, color=colors)
#         ax.set_title(
#             f"[{model_short}] {dataset_name} Predictive Importance "
#             f"(AUC={cv_scores.mean():.3f} ± {cv_scores.std():.3f})"
#         )
#         ax.set_xlabel("Coefficient (positive = predicts stereotype preference)")
#         ax.axvline(x=0, color="black", linewidth=0.8)
#         plt.tight_layout()
#         save_fig(output_dir, model_short, dataset_name, f"fig9{fig_prefix_letter}", "predictive_importance")

#         pred_rows = [{
#             "model": model_short, "dataset": dataset_name,
#             "analysis": "predictive_stereotype_preference",
#             "layer": analysis_layer,
#             "auc_mean": float(cv_scores.mean()),
#             "auc_std": float(cv_scores.std()),
#             "n_samples": int(len(y)),
#             "n_positive": int(sum(y)),
#         }]
#         for emotion, coef in zip(sorted(emotions), coefs):
#             pred_rows.append({
#                 "model": model_short, "dataset": dataset_name,
#                 "analysis": "predictive_feature_importance",
#                 "layer": analysis_layer,
#                 "emotion": emotion, "coefficient": float(coef),
#             })
#         save_stats_csv(pred_rows, output_dir, model_short, f"{dataset_name}_predictive")
#     else:
#         print(f"  [skip predictive] insufficient class variation "
#               f"(N={len(y)}, classes={set(y) if len(y)>0 else '∅'})")

#     return df, stat_rows


# # ============================================================
# # StereoSet & GenAssocBias wrappers
# # ============================================================

# def analyze_stereoset(data, output_dir, model_short):
#     return _three_condition_analysis(
#         data, output_dir, model_short, dataset_name="stereoset",
#         fig_prefix_letter="",
#     )


# def analyze_genassocbias(data, output_dir, model_short):
#     return _three_condition_analysis(
#         data, output_dir, model_short, dataset_name="genassocbias",
#         fig_prefix_letter="g",  # so files don't collide with stereoset's fig1/fig2 if combined
#     )


# # ============================================================
# # BBQ Analysis (split by ambig/disambig, robust to 3 response types)
# # ============================================================

# def _bbq_response_types_for(cond_name):
#     """Return (all_types, focus_type, baseline_type) for ambig/disambig."""
#     if cond_name == "ambig":
#         return ["correct", "stereotyped_guess", "others"], "stereotyped_guess", "correct"
#     else:
#         return ["correct", "incorrect", "others"], "incorrect", "correct"


# def analyze_bbq_enhanced(data, output_dir, model_short):
#     results = data["results"]
#     emotions = data["emotions"]
#     layers = data["target_layers"]
#     analysis_layer = layers[len(layers) * 2 // 3]

#     print(f"\n[{model_short}] ENHANCED BBQ ANALYSIS (layer {analysis_layer})")

#     # Split by ambig / disambig.
#     ambig_results = [r for r in results if r.get("is_ambiguous", False)
#                      or ("ambig" in r.get("context_condition", "").lower()
#                          and "disambig" not in r.get("context_condition", "").lower())]
#     disambig_results = [r for r in results if r not in ambig_results]

#     # ---- Response-type distribution (combined CSV) ----
#     from collections import Counter
#     dist_rows = []
#     for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
#         rt_counts = Counter(r["response_type"] for r in subset)
#         for rt, count in rt_counts.items():
#             dist_rows.append({
#                 "model": model_short, "context_condition": cond_name,
#                 "response_type": rt, "count": int(count),
#                 "pct": 100 * count / max(1, len(subset)),
#             })
#     save_stats_csv(dist_rows, output_dir, model_short, "bbq_response_distribution")

#     # ---- Distribution figures (per condition) ----
#     for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
#         if len(subset) < 5:
#             continue
#         all_types, _, _ = _bbq_response_types_for(cond_name)
#         rt_counts = Counter(r["response_type"] for r in subset)
#         counts = [rt_counts.get(rt, 0) for rt in all_types]
#         pcts = [100 * c / max(1, len(subset)) for c in counts]

#         fig, ax = plt.subplots(figsize=(8, 5))
#         bars = ax.bar(all_types, pcts,
#                       color=["#388e3c", "#d32f2f", "#9e9e9e"])
#         for bar, c, p in zip(bars, counts, pcts):
#             ax.text(bar.get_x() + bar.get_width()/2, p + 0.5,
#                     f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
#         ax.set_ylabel("% of responses")
#         ax.set_title(f"[{model_short}] BBQ-{cond_name}: Response-Type Distribution (N={len(subset)})")
#         ax.set_ylim(0, max(pcts) * 1.2 if pcts else 100)
#         plt.tight_layout()
#         save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig5", "response_distribution")

#     # ---- Per-condition emotion analyses ----
#     for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
#         if len(subset) < 5:
#             print(f"  [skip] BBQ-{cond_name}: only {len(subset)} examples")
#             continue

#         all_types, focus_label, baseline_label = _bbq_response_types_for(cond_name)

#         # Build long-form dataframe.
#         rows = []
#         for r in subset:
#             proj = r["emotion_projections_at_question"].get(analysis_layer, {})
#             for emotion, value in proj.items():
#                 rows.append({
#                     "category": r["category"],
#                     "response_type": r["response_type"],
#                     "emotion": emotion,
#                     "activation": value,
#                 })
#         df = pd.DataFrame(rows)
#         if df.empty:
#             continue

#         # ---- Bar plot by response type ----
#         fig, ax = plt.subplots(figsize=(14, 6))
#         pivot = df.groupby(["emotion", "response_type"])["activation"].mean().reset_index()
#         pivot_wide = pivot.pivot(index="emotion", columns="response_type", values="activation")
#         # Order columns consistently
#         ordered_cols = [c for c in all_types if c in pivot_wide.columns]
#         pivot_wide = pivot_wide[ordered_cols]
#         pivot_wide.plot(kind="bar", ax=ax, width=0.8)
#         ax.set_title(f"[{model_short}] BBQ-{cond_name}: Mean Emotion Activation by Response Type "
#                      f"(Layer {analysis_layer})")
#         ax.set_ylabel("Mean Projection")
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()
#         save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig5b", "by_response_type")

#         # ---- Heatmap: focus - baseline by category ----
#         if (focus_label in df["response_type"].values
#                 and baseline_label in df["response_type"].values):
#             cat_diffs = {}
#             for cat in df["category"].unique():
#                 sub = df[df["category"] == cat]
#                 fcs = sub[sub["response_type"] == focus_label]
#                 base = sub[sub["response_type"] == baseline_label]
#                 # Relaxed threshold
#                 if len(fcs) >= 3 and len(base) >= 3:
#                     cat_diffs[cat] = {}
#                     for emotion in emotions:
#                         fm = fcs[fcs["emotion"] == emotion]["activation"].mean()
#                         bm = base[base["emotion"] == emotion]["activation"].mean()
#                         cat_diffs[cat][emotion] = fm - bm

#             # Always include OVERALL row
#             overall = {}
#             for emotion in emotions:
#                 fv = df[(df["response_type"] == focus_label) & (df["emotion"] == emotion)]["activation"].mean()
#                 bv = df[(df["response_type"] == baseline_label) & (df["emotion"] == emotion)]["activation"].mean()
#                 overall[emotion] = fv - bv
#             cat_diffs["OVERALL"] = overall

#             diff_df = pd.DataFrame(cat_diffs).T
#             fig, ax = plt.subplots(figsize=(16, max(6, len(cat_diffs) * 0.55)))
#             sns.heatmap(diff_df, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".2f",
#                         linewidths=0.5, annot_kws={"fontsize": 7},
#                         cbar_kws={"label": f"{focus_label} − {baseline_label}"})
#             ax.set_title(f"[{model_short}] BBQ-{cond_name}: {focus_label} − {baseline_label} by Category")
#             plt.tight_layout()
#             save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig6", "heatmap")
#         else:
#             print(f"  [skip heatmap] BBQ-{cond_name}: missing one of "
#                   f"{focus_label!r}/{baseline_label!r} in response_type")

#         # ---- Layer-wise top 5 ----
#         fcs_emotions_diffs = {}
#         for emotion in emotions:
#             diffs_per_layer = []
#             for layer in layers:
#                 fv, bv = [], []
#                 for r in subset:
#                     proj = r["emotion_projections_at_question"].get(layer, {})
#                     if emotion in proj:
#                         if r["response_type"] == focus_label:
#                             fv.append(proj[emotion])
#                         elif r["response_type"] == baseline_label:
#                             bv.append(proj[emotion])
#                 if fv and bv:
#                     diffs_per_layer.append(np.mean(fv) - np.mean(bv))
#                 else:
#                     diffs_per_layer.append(0)
#             fcs_emotions_diffs[emotion] = diffs_per_layer

#         ref_idx = min(len(layers) * 2 // 3, len(layers) - 1)
#         top5 = sorted(
#             fcs_emotions_diffs.keys(),
#             key=lambda e: abs(fcs_emotions_diffs[e][ref_idx]) if fcs_emotions_diffs[e] else 0,
#             reverse=True,
#         )[:5]
#         if top5 and any(any(v != 0 for v in fcs_emotions_diffs[e]) for e in top5):
#             fig, ax = plt.subplots(figsize=(12, 6))
#             for emotion in top5:
#                 ax.plot(layers, fcs_emotions_diffs[emotion],
#                         marker="o", label=emotion, linewidth=2)
#             ax.set_xlabel("Layer")
#             ax.set_ylabel(f"{focus_label} − {baseline_label}")
#             ax.set_title(f"[{model_short}] BBQ-{cond_name}: Layer-wise Top 5 Emotions")
#             ax.legend()
#             ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
#             plt.tight_layout()
#             save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig4", "layerwise")

#         # ---- Predictive analysis (focus_label vs baseline_label) ----
#         X_rows, y = [], []
#         for r in subset:
#             proj = r["emotion_projections_at_question"].get(analysis_layer, {})
#             if len(proj) == len(emotions):
#                 if r["response_type"] == focus_label:
#                     X_rows.append([proj[e] for e in sorted(emotions)])
#                     y.append(1)
#                 elif r["response_type"] == baseline_label:
#                     X_rows.append([proj[e] for e in sorted(emotions)])
#                     y.append(0)
#         X = np.array(X_rows)
#         y = np.array(y)

#         if len(X) > 20 and len(set(y)) == 2:
#             clf = LogisticRegression(max_iter=1000, random_state=42)
#             cv = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
#             clf.fit(X, y)
#             coefs = clf.coef_[0]
#             importance = pd.Series(coefs, index=sorted(emotions)).sort_values()

#             fig, ax = plt.subplots(figsize=(10, 10))
#             colors = ["#d32f2f" if v > 0 else "#1976d2" for v in importance.values]
#             importance.plot(kind="barh", ax=ax, color=colors)
#             ax.set_title(f"[{model_short}] BBQ-{cond_name} Predictive "
#                          f"(AUC={cv.mean():.3f} ± {cv.std():.3f})\n"
#                          f"+coef ⇒ predicts {focus_label}")
#             ax.axvline(x=0, color="black", linewidth=0.8)
#             plt.tight_layout()
#             save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig9", "predictive")

#             pred_rows = [{
#                 "model": model_short, "context": cond_name,
#                 "analysis": "predictive",
#                 "auc_mean": float(cv.mean()), "auc_std": float(cv.std()),
#                 "n_samples": int(len(y)),
#                 "focus_class": focus_label, "baseline_class": baseline_label,
#             }]
#             for e, c in zip(sorted(emotions), coefs):
#                 pred_rows.append({
#                     "model": model_short, "context": cond_name,
#                     "analysis": "feature_importance",
#                     "emotion": e, "coefficient": float(c),
#                 })
#             save_stats_csv(pred_rows, output_dir, model_short, f"bbq_{cond_name}_predictive")

#         # ---- t-tests with effect sizes ----
#         ttest_rows = []
#         for emotion in sorted(emotions):
#             fv = df[(df["response_type"] == focus_label) & (df["emotion"] == emotion)]["activation"].values
#             bv = df[(df["response_type"] == baseline_label) & (df["emotion"] == emotion)]["activation"].values
#             if len(fv) >= 3 and len(bv) >= 3:
#                 t, p = stats.ttest_ind(fv, bv, equal_var=False)
#                 pooled = np.sqrt((np.var(fv) + np.var(bv)) / 2)
#                 d = (np.mean(fv) - np.mean(bv)) / pooled if pooled > 0 else 0
#                 ttest_rows.append({
#                     "model": model_short, "context": cond_name, "emotion": emotion,
#                     "focus_class": focus_label, "baseline_class": baseline_label,
#                     "focus_mean": float(np.mean(fv)),
#                     "baseline_mean": float(np.mean(bv)),
#                     "diff": float(np.mean(fv) - np.mean(bv)),
#                     "cohen_d": float(d),
#                     "t_stat": float(t), "p_value": float(p),
#                     "significance": _sig(p),
#                     "n_focus": int(len(fv)), "n_baseline": int(len(bv)),
#                 })
#         save_stats_csv(ttest_rows, output_dir, model_short, f"bbq_{cond_name}_ttests")


# # ============================================================
# # BOLD Analysis
# # ============================================================

# def analyze_bold(data, output_dir, model_short):
#     results = data["results"]
#     emotions = data["emotions"]
#     layers = data["target_layers"]
#     analysis_layer = layers[len(layers) * 2 // 3]

#     print(f"\n[{model_short}] BOLD ANALYSIS (layer {analysis_layer})")

#     rows = []
#     for r in results:
#         proj = r["emotion_projections_after_generation"].get(analysis_layer, {})
#         prompt_proj = r["emotion_projections_at_prompt"].get(analysis_layer, {})
#         for emotion in emotions:
#             if emotion in proj and emotion in prompt_proj:
#                 rows.append({
#                     "domain": r["domain"], "category": r["category"], "emotion": emotion,
#                     "activation_after": proj[emotion],
#                     "activation_before": prompt_proj[emotion],
#                     "activation_change": proj[emotion] - prompt_proj[emotion],
#                 })
#     df = pd.DataFrame(rows)
#     if df.empty:
#         return

#     # ---- fig7: by domain ----
#     fig, ax = plt.subplots(figsize=(14, 6))
#     pivot = df.groupby(["emotion", "domain"])["activation_after"].mean().reset_index()
#     pivot_wide = pivot.pivot(index="emotion", columns="domain", values="activation_after")
#     pivot_wide.plot(kind="bar", ax=ax, width=0.8)
#     ax.set_title(f"[{model_short}] BOLD: Mean Emotion Activation After Generation by Domain "
#                  f"(Layer {analysis_layer})")
#     ax.set_ylabel("Mean Projection")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     save_fig(output_dir, model_short, "bold", "fig7", "by_domain")

#     # ---- fig8: emotion change heatmap ----
#     fig, ax = plt.subplots(figsize=(14, 8))
#     change_by_domain = df.groupby(["emotion", "domain"])["activation_change"].mean().reset_index()
#     pivot_change = change_by_domain.pivot(index="emotion", columns="domain", values="activation_change")
#     sns.heatmap(pivot_change, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".3f",
#                 linewidths=0.5, annot_kws={"fontsize": 7},
#                 cbar_kws={"label": "Activation Change (After − Before)"})
#     ax.set_title(f"[{model_short}] BOLD: Emotion Activation Change During Generation")
#     plt.tight_layout()
#     save_fig(output_dir, model_short, "bold", "fig8", "emotion_change")

#     # ---- per-domain stats CSV ----
#     stat_rows = []
#     for domain in sorted(df["domain"].unique()):
#         sub = df[df["domain"] == domain]
#         for emotion in sorted(emotions):
#             es = sub[sub["emotion"] == emotion]
#             if len(es) > 0:
#                 stat_rows.append({
#                     "model": model_short, "dataset": "bold",
#                     "domain": domain, "layer": analysis_layer, "emotion": emotion,
#                     "mean_after_generation": float(es["activation_after"].mean()),
#                     "mean_before_generation": float(es["activation_before"].mean()),
#                     "mean_change": float(es["activation_change"].mean()),
#                     "n": int(len(es)),
#                 })
#     save_stats_csv(stat_rows, output_dir, model_short, "bold_domain_stats")

#     # ---- layer-wise top variable emotions ----
#     domain_variance = {}
#     for emotion in emotions:
#         means = df[df["emotion"] == emotion].groupby("domain")["activation_after"].mean()
#         domain_variance[emotion] = means.var() if len(means) > 1 else 0
#     top_var = sorted(domain_variance.keys(), key=lambda e: -domain_variance[e])[:5]

#     if top_var:
#         fig, ax = plt.subplots(figsize=(12, 6))
#         for emotion in top_var:
#             layer_means = []
#             for layer in layers:
#                 vals = []
#                 for r in results:
#                     proj = r["emotion_projections_after_generation"].get(layer, {})
#                     if emotion in proj:
#                         vals.append(proj[emotion])
#                 layer_means.append(np.mean(vals) if vals else 0)
#             ax.plot(layers, layer_means, marker="o", label=emotion, linewidth=2)
#         ax.set_xlabel("Layer")
#         ax.set_ylabel("Mean Emotion Activation")
#         ax.set_title(f"[{model_short}] BOLD: Layer-wise Top 5 Domain-Variable Emotions")
#         ax.legend()
#         plt.tight_layout()
#         save_fig(output_dir, model_short, "bold", "fig4", "layerwise")

#     # ---- category heatmap (top 8) ----
#     top_cats = df["category"].value_counts().head(8).index.tolist()
#     if top_cats:
#         cat_df = df[df["category"].isin(top_cats)]
#         pivot = cat_df.groupby(["emotion", "category"])["activation_after"].mean().reset_index()
#         pivot_wide = pivot.pivot(index="emotion", columns="category", values="activation_after")
#         fig, ax = plt.subplots(figsize=(14, 10))
#         sns.heatmap(pivot_wide, center=0, cmap="RdBu_r", ax=ax, annot=False,
#                     cbar_kws={"label": "Mean Activation"})
#         ax.set_title(f"[{model_short}] BOLD: Emotion Activation by Top 8 Categories")
#         plt.tight_layout()
#         save_fig(output_dir, model_short, "bold", "fig6", "category")

#     # ---- ANOVA across domains ----
#     anova_rows = []
#     for emotion in sorted(emotions):
#         groups, names = [], []
#         for domain in df["domain"].unique():
#             vals = df[(df["domain"] == domain) & (df["emotion"] == emotion)]["activation_after"].values
#             if len(vals) > 5:
#                 groups.append(vals); names.append(domain)
#         if len(groups) >= 2:
#             try:
#                 f_stat, p_val = stats.f_oneway(*groups)
#                 anova_rows.append({
#                     "model": model_short, "dataset": "bold", "emotion": emotion,
#                     "n_domains": len(groups),
#                     "f_stat": float(f_stat), "p_value": float(p_val),
#                     "significance": _sig(p_val),
#                     "max_mean": float(max(np.mean(g) for g in groups)),
#                     "min_mean": float(min(np.mean(g) for g in groups)),
#                 })
#             except Exception:
#                 pass
#     save_stats_csv(anova_rows, output_dir, model_short, "bold_anova")


# # ============================================================
# # Cross-Dataset Comparison
# # ============================================================

# def analyze_cross_dataset(stereoset_data, genassoc_data, bbq_data, bold_data,
#                           output_dir, model_short):
#     print(f"\n[{model_short}] CROSS-DATASET ANALYSIS")

#     dataset_diffs = {}

#     def _three_cond_diff(data, label_in_plot):
#         results = data["results"]
#         emotions = data["emotions"]
#         layers = data["target_layers"]
#         analysis_layer = layers[len(layers) * 2 // 3]
#         out = {}
#         for emotion in emotions:
#             sv, av = [], []
#             for r in results:
#                 s_proj = r["emotion_projections"]["stereotype"].get(analysis_layer, {})
#                 a_proj = r["emotion_projections"]["anti_stereotype"].get(analysis_layer, {})
#                 if emotion in s_proj and emotion in a_proj:
#                     sv.append(s_proj[emotion]); av.append(a_proj[emotion])
#             if sv:
#                 out[emotion] = np.mean(sv) - np.mean(av)
#         dataset_diffs[label_in_plot] = out

#     if stereoset_data:
#         _three_cond_diff(stereoset_data, "StereoSet\n(stereo − anti)")
#     if genassoc_data:
#         _three_cond_diff(genassoc_data, "GenAssocBias\n(stereo − anti)")

#     if bbq_data:
#         results = bbq_data["results"]
#         emotions = bbq_data["emotions"]
#         layers = bbq_data["target_layers"]
#         analysis_layer = layers[len(layers) * 2 // 3]
#         # Use ambig only: stereotyped_guess − correct
#         out = {}
#         for emotion in emotions:
#             sg, co = [], []
#             for r in results:
#                 if not r.get("is_ambiguous", False):
#                     continue
#                 proj = r["emotion_projections_at_question"].get(analysis_layer, {})
#                 if emotion in proj:
#                     if r["response_type"] == "stereotyped_guess":
#                         sg.append(proj[emotion])
#                     elif r["response_type"] == "correct":
#                         co.append(proj[emotion])
#             if sg and co:
#                 out[emotion] = np.mean(sg) - np.mean(co)
#         if out:
#             dataset_diffs["BBQ-ambig\n(stereo guess − correct)"] = out

#     if bold_data:
#         results = bold_data["results"]
#         emotions = bold_data["emotions"]
#         layers = bold_data["target_layers"]
#         analysis_layer = layers[len(layers) * 2 // 3]
#         out = {}
#         for emotion in emotions:
#             rv, pv = [], []
#             for r in results:
#                 proj = r["emotion_projections_after_generation"].get(analysis_layer, {})
#                 if emotion in proj:
#                     if r["domain"] == "race":
#                         rv.append(proj[emotion])
#                     elif r["domain"] == "profession":
#                         pv.append(proj[emotion])
#             if rv and pv:
#                 out[emotion] = np.mean(rv) - np.mean(pv)
#         if out:
#             dataset_diffs["BOLD\n(race − profession)"] = out

#     if not dataset_diffs:
#         print("  No datasets available.")
#         return

#     all_emotions = sorted({e for d in dataset_diffs.values() for e in d.keys()})
#     df = pd.DataFrame(index=all_emotions)
#     for ds_name, diffs in dataset_diffs.items():
#         df[ds_name] = pd.Series(diffs)
#     df = df.fillna(0)
#     df = df.assign(_mean=df.mean(axis=1)).sort_values("_mean").drop(columns=["_mean"])

#     # ---- fig10: grouped bars ----
#     fig, ax = plt.subplots(figsize=(14, 14))
#     n_emotions = len(df)
#     n_datasets = len(df.columns)
#     bar_height = 0.8 / max(1, n_datasets)
#     y_positions = np.arange(n_emotions)
#     palette = ["#d32f2f", "#1976d2", "#388e3c", "#f57c00", "#7b1fa2"]
#     for i, col in enumerate(df.columns):
#         offset = (i - n_datasets / 2 + 0.5) * bar_height
#         ax.barh(y_positions + offset, df[col], height=bar_height,
#                 label=col, color=palette[i % len(palette)], alpha=0.85)
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(df.index, fontsize=10)
#     ax.set_xlabel("Mean Activation Difference")
#     ax.set_title(f"[{model_short}] Emotion Activation Difference Across All Datasets")
#     ax.axvline(x=0, color="black", linewidth=0.8)
#     ax.legend(fontsize=10, loc="lower right")
#     plt.tight_layout()
#     save_fig(output_dir, model_short, "cross", "fig10", "comparison")

#     # ---- fig11: heatmap ----
#     fig, ax = plt.subplots(figsize=(10, 14))
#     sns.heatmap(df, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".2f",
#                 linewidths=0.5, cbar_kws={"label": "Activation Difference"},
#                 annot_kws={"fontsize": 8})
#     ax.set_title(f"[{model_short}] Cross-Dataset Heatmap")
#     ax.set_ylabel("Emotion")
#     plt.tight_layout()
#     save_fig(output_dir, model_short, "cross", "fig11", "heatmap")

#     # ---- CSVs ----
#     csv_rows = []
#     for emotion in df.index:
#         row = {"model": model_short, "emotion": emotion}
#         for col in df.columns:
#             row[col.replace("\n", " ").replace("−", "minus")] = float(df.loc[emotion, col])
#         csv_rows.append(row)
#     save_stats_csv(csv_rows, output_dir, model_short, "cross_dataset_diffs")

#     corr_rows = []
#     cols = df.columns.tolist()
#     print("\n  Cross-dataset correlations:")
#     for i in range(len(cols)):
#         for j in range(i + 1, len(cols)):
#             valid = df[[cols[i], cols[j]]].dropna()
#             if len(valid) > 3:
#                 r, p = stats.pearsonr(valid[cols[i]], valid[cols[j]])
#                 ni = cols[i].replace("\n", " "); nj = cols[j].replace("\n", " ")
#                 print(f"    {ni} <-> {nj}: r={r:.3f}, p={p:.4f}")
#                 corr_rows.append({
#                     "model": model_short, "dataset_1": ni, "dataset_2": nj,
#                     "pearson_r": float(r), "p_value": float(p),
#                     "n_emotions": int(len(valid)),
#                 })
#     save_stats_csv(corr_rows, output_dir, model_short, "cross_dataset_correlations")


# # ============================================================
# # Main
# # ============================================================

# def main():
#     global ALL_STATS
#     ALL_STATS = []

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_slug", type=str, required=True,
#                         help="Long HF-style slug, e.g. google_gemma_2_2b_it")
#     parser.add_argument("--model_short", type=str, default="",
#                         help="Short identifier for filenames; auto-derived if omitted")
#     parser.add_argument("--stereoset", type=str, required=True)
#     parser.add_argument("--genassocbias", type=str, default="")
#     parser.add_argument("--bbq", type=str, required=True)
#     parser.add_argument("--bold", type=str, required=True)
#     parser.add_argument("--output", type=str, default="outputs/figures/")
#     args = parser.parse_args()

#     os.makedirs(args.output, exist_ok=True)
#     plt.style.use("seaborn-v0_8-whitegrid")

#     model_short = args.model_short.strip() or shorten_slug(args.model_slug)
#     print(f"Model slug : {args.model_slug}")
#     print(f"Model short: {model_short}")

#     stereoset_data = bbq_data = bold_data = genassoc_data = None

#     if os.path.exists(args.stereoset):
#         stereoset_data = load_results(args.stereoset)
#         analyze_stereoset(stereoset_data, args.output, model_short)

#     if args.genassocbias and os.path.exists(args.genassocbias):
#         genassoc_data = load_results(args.genassocbias)
#         analyze_genassocbias(genassoc_data, args.output, model_short)

#     if os.path.exists(args.bbq):
#         bbq_data = load_results(args.bbq)
#         analyze_bbq_enhanced(bbq_data, args.output, model_short)

#     if os.path.exists(args.bold):
#         bold_data = load_results(args.bold)
#         analyze_bold(bold_data, args.output, model_short)

#     analyze_cross_dataset(stereoset_data, genassoc_data, bbq_data, bold_data,
#                           args.output, model_short)

#     if ALL_STATS:
#         unified_path = os.path.join(args.output, f"{model_short}_ALL_STATS.csv")
#         pd.DataFrame(ALL_STATS).to_csv(unified_path, index=False)
#         print(f"\n{'='*70}\nUNIFIED STATS FILE: {unified_path}\n"
#               f"  Total rows: {len(ALL_STATS)}\n{'='*70}")

#     print(f"\n[{model_short}] All figures & CSVs saved to {args.output}")


# if __name__ == "__main__":
#     main()



"""
analyze_results.py - Statistical analysis and figure generation.

Filename convention:
    {model_short}_{dataset}_{figtype}_{description}.pdf
e.g. gemma2_2b_stereoset_fig2_emotion_diff.pdf
     gemma2_2b_bbq_ambig_fig5_response_distribution.pdf
     gemma2_2b_genassocbias_fig9_predictive.pdf
     gemma2_2b_cross_fig10_comparison.pdf
"""

import os
import re
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler # Added for stability


# ============================================================
# Helpers
# ============================================================

def load_results(path):
    return torch.load(path, weights_only=False)


# Global stats accumulator across all analyses
ALL_STATS = []


def add_stats(rows, name):
    for r in rows:
        r["section"] = name
        ALL_STATS.append(r)


def shorten_slug(slug: str) -> str:
    """Convert HF-style slug into a short filename-friendly identifier.

    Examples:
      google_gemma_2_2b_it                  -> gemma2_2b
      meta_llama_llama_3.2_3b_instruct      -> llama32_3b
      mistralai_mistral_7b_instruct_v0.3    -> mistral_7b
      qwen_qwen2.5_7b_instruct              -> qwen25_7b
      meta_llama_meta_llama_3.1_8b_instruct -> llama31_8b
    """
    s = slug.lower()
    prefixes = ["google_", "meta_llama_", "mistralai_", "qwen_",
                "anthropic_", "microsoft_", "openai_"]
    suffixes = ["_it", "_instruct", "_chat", "_hf", "_base"]
    version_re = re.compile(r"_v\d+(?:[._]\d+)*$")

    # Strip ORG prefix exactly once (don't re-strip — model name may legitimately
    # start with the same string, e.g. "meta_llama_meta_llama_3.1_8b_instruct").
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
            break

    # Strip suffixes + version markers iteratively (order between them matters).
    changed = True
    while changed:
        changed = False
        for sfx in suffixes:
            if s.endswith(sfx):
                s = s[:-len(sfx)]; changed = True
        new = version_re.sub("", s)
        if new != s:
            s = new; changed = True

    # Compact version numbers and normalize "meta_llama" -> "llama":
    #   gemma_2_2b -> gemma2_2b
    #   llama_3.2_3b -> llama32_3b
    #   meta_llama_3.1_8b -> llama31_8b
    #   qwen2.5_7b -> qwen25_7b
    s = re.sub(
        r"^(?:meta_)?(gemma|llama|qwen|mistral|phi|mixtral)_?(\d+(?:\.\d+)?)_",
        lambda m: f"{m.group(1)}{m.group(2).replace('.', '')}_",
        s,
    )
    return s


def fig_path(output_dir, model_short, dataset, figtype, description, ext="pdf"):
    """Build a figure path with the new convention."""
    fname = f"{model_short}_{dataset}_{figtype}_{description}.{ext}"
    return os.path.join(output_dir, fname)


def save_fig(output_dir, model_short, dataset, figtype, description):
    """Save current matplotlib figure as both PDF and PNG."""
    plt.savefig(fig_path(output_dir, model_short, dataset, figtype, description, "pdf"), dpi=150)
    plt.savefig(fig_path(output_dir, model_short, dataset, figtype, description, "png"), dpi=150)
    plt.close()


def save_stats_csv(rows, output_dir, model_short, name):
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{model_short}_stats_{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Stats saved: {path}")
    add_stats(rows, name)


def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ============================================================
# Generic 3-condition analysis (used by StereoSet AND GenAssocBias)
# ============================================================

def _three_condition_analysis(
    data, output_dir, model_short, dataset_name,
    fig_prefix_letter="",
):
    """
    Produce the standard 5-figure suite for any dataset that has
    {stereotype, anti_stereotype, unrelated} conditions:
      fig1  - Mean activation by condition (grouped bars)
      fig2  - stereotype - anti_stereotype difference (horizontal bars)
      fig3  - per-bias-type difference (2x2 grid)
      fig4  - layer-wise top-5 emotions
      fig9  - predictive importance (LR coefficients)
    """
    results = data["results"]
    emotions = data["emotions"]
    layers = data["target_layers"]
    analysis_layer = layers[len(layers) * 2 // 3]

    print(f"\n[{model_short}] ANALYSIS: {dataset_name} (layer {analysis_layer})")

    rows = []
    for r in results:
        for condition in ["stereotype", "anti_stereotype", "unrelated"]:
            proj = r["emotion_projections"][condition].get(analysis_layer, {})
            for emotion, value in proj.items():
                rows.append({
                    "id": r.get("id"),
                    "bias_type": r.get("bias_type", "unknown"),
                    "condition": condition,
                    "prefers_stereotype": r.get("prefers_stereotype"),
                    "emotion": emotion,
                    "activation": value,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  [skip] empty dataframe for {dataset_name}")
        return

    # ---- fig1: by condition ----
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df.groupby(["emotion", "condition"])["activation"].mean().reset_index()
    pivot_wide = pivot.pivot(index="emotion", columns="condition", values="activation")
    pivot_wide = pivot_wide.reindex(columns=["stereotype", "anti_stereotype", "unrelated"])
    pivot_wide.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(f"[{model_short}] {dataset_name}: Mean Emotion Activation by Condition (Layer {analysis_layer})")
    ax.set_ylabel("Mean Projection onto Emotion Vector")
    ax.set_xlabel("")
    ax.legend(title="Condition")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig(output_dir, model_short, dataset_name, f"fig1{fig_prefix_letter}", "emotion_by_condition")

    # ---- fig2: stereo - anti diff ----
    s_means = df[df["condition"] == "stereotype"].groupby("emotion")["activation"].mean()
    a_means = df[df["condition"] == "anti_stereotype"].groupby("emotion")["activation"].mean()
    diff = (s_means - a_means).sort_values()

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in diff.values]
    diff.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"[{model_short}] {dataset_name}: Stereotype − Anti-stereotype")
    ax.set_xlabel("Mean Activation Difference")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_fig(output_dir, model_short, dataset_name, f"fig2{fig_prefix_letter}", "emotion_diff")

    # ---- fig3: per-bias-type 2x2 ----
    bias_types = [b for b in df["bias_type"].unique() if pd.notna(b)]
    n_types = min(4, len(bias_types))
    if n_types > 0:
        n_rows = max(1, int(np.ceil(n_types / 2)))
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), squeeze=False)
        for idx in range(n_types):
            bt = bias_types[idx]
            ax = axes[idx // 2][idx % 2]
            subset = df[df["bias_type"] == bt]
            sm = subset[subset["condition"] == "stereotype"].groupby("emotion")["activation"].mean()
            am = subset[subset["condition"] == "anti_stereotype"].groupby("emotion")["activation"].mean()
            diff_bt = (sm - am).sort_values()
            colors = ["#d32f2f" if v > 0 else "#1976d2" for v in diff_bt.values]
            diff_bt.plot(kind="barh", ax=ax, color=colors)
            ax.set_title(f"Bias Type: {bt}", fontsize=12)
            ax.axvline(x=0, color="black", linewidth=0.8)
        # Hide unused axes
        for idx in range(n_types, n_rows * 2):
            axes[idx // 2][idx % 2].axis("off")
        plt.suptitle(f"[{model_short}] {dataset_name}: Difference by Bias Type", fontsize=14)
        plt.tight_layout()
        save_fig(output_dir, model_short, dataset_name, f"fig3{fig_prefix_letter}", "by_bias_type")

    # ---- t-tests ----
    stat_rows = []
    for emotion in sorted(emotions):
        s_vals = df[(df["condition"] == "stereotype") & (df["emotion"] == emotion)]["activation"].values
        a_vals = df[(df["condition"] == "anti_stereotype") & (df["emotion"] == emotion)]["activation"].values
        if len(s_vals) > 0 and len(a_vals) > 0:
            n = min(len(s_vals), len(a_vals))
            t_stat, p_val = stats.ttest_rel(s_vals[:n], a_vals[:n])
            stat_rows.append({
                "model": model_short, "dataset": dataset_name,
                "analysis": "stereotype_vs_anti_stereotype",
                "layer": analysis_layer, "emotion": emotion,
                "stereo_mean": float(np.mean(s_vals)),
                "anti_mean": float(np.mean(a_vals)),
                "diff": float(np.mean(s_vals) - np.mean(a_vals)),
                "t_stat": float(t_stat), "p_value": float(p_val),
                "significance": _sig(p_val),
                "n_stereo": int(len(s_vals)), "n_anti": int(len(a_vals)),
            })
    save_stats_csv(stat_rows, output_dir, model_short, f"{dataset_name}_ttests")

    # ---- fig4: layer-wise top 5 ----
    sig_sorted = sorted(
        [(r["emotion"], r["diff"], r["p_value"]) for r in stat_rows if r["p_value"] < 0.05],
        key=lambda x: abs(x[1]), reverse=True,
    )
    top_emotions = [e for e, _, _ in sig_sorted[:5]]
    if not top_emotions:
        # Fallback to top-5 by absolute diff regardless of significance
        top_emotions = sorted(stat_rows, key=lambda r: abs(r["diff"]), reverse=True)[:5]
        top_emotions = [r["emotion"] for r in top_emotions]
    if top_emotions:
        fig, ax = plt.subplots(figsize=(12, 6))
        for emotion in top_emotions:
            diffs_by_layer = []
            for layer in layers:
                s_vals, a_vals = [], []
                for r in results:
                    s_proj = r["emotion_projections"]["stereotype"].get(layer, {})
                    a_proj = r["emotion_projections"]["anti_stereotype"].get(layer, {})
                    if emotion in s_proj and emotion in a_proj:
                        s_vals.append(s_proj[emotion])
                        a_vals.append(a_proj[emotion])
                diffs_by_layer.append(np.mean(s_vals) - np.mean(a_vals) if s_vals else 0)
            ax.plot(layers, diffs_by_layer, marker="o", label=emotion, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Activation Difference (Stereo − Anti)")
        ax.set_title(f"[{model_short}] {dataset_name}: Layer-wise (Top 5 emotions)")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        plt.tight_layout()
        save_fig(output_dir, model_short, dataset_name, f"fig4{fig_prefix_letter}", "layerwise")

    # ---- fig9: predictive importance ----
    X_rows, y = [], []
    for r in results:
        proj = r["emotion_projections"]["stereotype"].get(analysis_layer, {})
        if len(proj) == len(emotions):
            X_rows.append([proj[e] for e in sorted(emotions)])
            y.append(1 if r.get("prefers_stereotype") else 0)
    X = np.array(X_rows)
    y = np.array(y)

    if len(X) >= 20 and len(set(y)) == 2:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        clf.fit(X, y)
        coefs = clf.coef_[0]
        importance = pd.Series(coefs, index=sorted(emotions)).sort_values()

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ["#d32f2f" if v > 0 else "#1976d2" for v in importance.values]
        importance.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(
            f"[{model_short}] {dataset_name} Predictive Importance "
            f"(AUC={cv_scores.mean():.3f} ± {cv_scores.std():.3f})"
        )
        ax.set_xlabel("Coefficient (positive = predicts stereotype preference)")
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        save_fig(output_dir, model_short, dataset_name, f"fig9{fig_prefix_letter}", "predictive_importance")

        pred_rows = [{
            "model": model_short, "dataset": dataset_name,
            "analysis": "predictive_stereotype_preference",
            "layer": analysis_layer,
            "auc_mean": float(cv_scores.mean()),
            "auc_std": float(cv_scores.std()),
            "n_samples": int(len(y)),
            "n_positive": int(sum(y)),
        }]
        for emotion, coef in zip(sorted(emotions), coefs):
            pred_rows.append({
                "model": model_short, "dataset": dataset_name,
                "analysis": "predictive_feature_importance",
                "layer": analysis_layer,
                "emotion": emotion, "coefficient": float(coef),
            })
        save_stats_csv(pred_rows, output_dir, model_short, f"{dataset_name}_predictive")
    else:
        print(f"  [skip predictive] insufficient class variation "
              f"(N={len(y)}, classes={set(y) if len(y)>0 else '∅'})")

    return df, stat_rows


# ============================================================
# StereoSet & GenAssocBias wrappers
# ============================================================

def analyze_stereoset(data, output_dir, model_short):
    return _three_condition_analysis(
        data, output_dir, model_short, dataset_name="stereoset",
        fig_prefix_letter="",
    )


def analyze_genassocbias(data, output_dir, model_short):
    return _three_condition_analysis(
        data, output_dir, model_short, dataset_name="genassocbias",
        fig_prefix_letter="g",  # so files don't collide with stereoset's fig1/fig2 if combined
    )


# ============================================================
# BBQ Analysis (split by ambig/disambig, robust to 3 response types)
# ============================================================

def _bbq_response_types_for(cond_name):
    """Return (all_types, focus_type, baseline_type) for ambig/disambig."""
    if cond_name == "ambig":
        return ["correct", "stereotyped_guess", "others"], "stereotyped_guess", "correct"
    else:
        return ["correct", "incorrect", "others"], "incorrect", "correct"


def analyze_bbq_enhanced(data, output_dir, model_short):
    results = data["results"]
    emotions = data["emotions"]
    layers = data["target_layers"]
    analysis_layer = layers[len(layers) * 2 // 3]

    print(f"\n[{model_short}] ENHANCED BBQ ANALYSIS (layer {analysis_layer})")

    # Split by ambig / disambig using the explicit condition key saved by the new probe_bbq.py
    ambig_results = [r for r in results if r.get("condition") == "ambig" or r.get("is_ambiguous") is True]
    disambig_results = [r for r in results if r.get("condition") == "disambig" or r.get("is_ambiguous") is False]

    # ---- Response-type distribution (combined CSV) ----
    from collections import Counter
    dist_rows = []
    for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
        rt_counts = Counter(r["response_type"] for r in subset)
        for rt, count in rt_counts.items():
            dist_rows.append({
                "model": model_short, "context_condition": cond_name,
                "response_type": rt, "count": int(count),
                "pct": 100 * count / max(1, len(subset)),
            })
    save_stats_csv(dist_rows, output_dir, model_short, "bbq_response_distribution")

    # ---- Distribution figures (per condition) ----
    for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
        if len(subset) < 1:
            continue
        all_types, _, _ = _bbq_response_types_for(cond_name)
        rt_counts = Counter(r["response_type"] for r in subset)
        counts = [rt_counts.get(rt, 0) for rt in all_types]
        pcts = [100 * c / max(1, len(subset)) for c in counts]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(all_types, pcts,
                      color=["#388e3c", "#d32f2f", "#9e9e9e"])
        for bar, c, p in zip(bars, counts, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.5,
                    f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("% of responses")
        ax.set_title(f"[{model_short}] BBQ-{cond_name}: Response-Type Distribution (N={len(subset)})")
        ax.set_ylim(0, max(pcts) * 1.2 if pcts else 100)
        plt.tight_layout()
        save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig5", "response_distribution")

    # ---- Per-condition emotion analyses ----
    for cond_name, subset in [("ambig", ambig_results), ("disambig", disambig_results)]:
        if len(subset) < 2:
            print(f"  [skip] BBQ-{cond_name}: only {len(subset)} examples")
            continue

        all_types, focus_label, baseline_label = _bbq_response_types_for(cond_name)

        # Build long-form dataframe.
        rows = []
        for r in subset:
            proj = r["emotion_projections_at_question"].get(analysis_layer, {})
            for emotion, value in proj.items():
                rows.append({
                    "category": r["category"],
                    "response_type": r["response_type"],
                    "emotion": emotion,
                    "activation": value,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        # ---- Bar plot by response type ----
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot = df.groupby(["emotion", "response_type"])["activation"].mean().reset_index()
        pivot_wide = pivot.pivot(index="emotion", columns="response_type", values="activation")
        # Order columns consistently
        ordered_cols = [c for c in all_types if c in pivot_wide.columns]
        pivot_wide = pivot_wide[ordered_cols]
        pivot_wide.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"[{model_short}] BBQ-{cond_name}: Mean Emotion Activation by Response Type "
                     f"(Layer {analysis_layer})")
        ax.set_ylabel("Mean Projection")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig5b", "by_response_type")

        # ---- Heatmap: focus - baseline by category ----
        if (focus_label in df["response_type"].values
                and baseline_label in df["response_type"].values):
            cat_diffs = {}
            for cat in df["category"].unique():
                sub = df[df["category"] == cat]
                fcs = sub[sub["response_type"] == focus_label]
                base = sub[sub["response_type"] == baseline_label]
                # Lowered threshold: generate heatmap even if only 1 example per response_type
                if len(fcs) >= 1 and len(base) >= 1:
                    cat_diffs[cat] = {}
                    for emotion in emotions:
                        fm = fcs[fcs["emotion"] == emotion]["activation"].mean()
                        bm = base[base["emotion"] == emotion]["activation"].mean()
                        cat_diffs[cat][emotion] = fm - bm

            # Always include OVERALL row
            overall = {}
            for emotion in emotions:
                fv = df[(df["response_type"] == focus_label) & (df["emotion"] == emotion)]["activation"].mean()
                bv = df[(df["response_type"] == baseline_label) & (df["emotion"] == emotion)]["activation"].mean()
                overall[emotion] = fv - bv
            cat_diffs["OVERALL"] = overall

            diff_df = pd.DataFrame(cat_diffs).T
            fig, ax = plt.subplots(figsize=(16, max(6, len(cat_diffs) * 0.55)))
            sns.heatmap(diff_df, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".2f",
                        linewidths=0.5, annot_kws={"fontsize": 7},
                        cbar_kws={"label": f"{focus_label} − {baseline_label}"})
            ax.set_title(f"[{model_short}] BBQ-{cond_name}: {focus_label} − {baseline_label} by Category")
            plt.tight_layout()
            save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig6", "heatmap")
        else:
            print(f"  [skip heatmap] BBQ-{cond_name}: missing one of "
                  f"{focus_label!r}/{baseline_label!r} in response_type")

        # ---- Layer-wise top 5 ----
        fcs_emotions_diffs = {}
        for emotion in emotions:
            diffs_per_layer = []
            for layer in layers:
                fv, bv = [], []
                for r in subset:
                    proj = r["emotion_projections_at_question"].get(layer, {})
                    if emotion in proj:
                        if r["response_type"] == focus_label:
                            fv.append(proj[emotion])
                        elif r["response_type"] == baseline_label:
                            bv.append(proj[emotion])
                if fv and bv:
                    diffs_per_layer.append(np.mean(fv) - np.mean(bv))
                else:
                    diffs_per_layer.append(0)
            fcs_emotions_diffs[emotion] = diffs_per_layer

        ref_idx = min(len(layers) * 2 // 3, len(layers) - 1)
        top5 = sorted(
            fcs_emotions_diffs.keys(),
            key=lambda e: abs(fcs_emotions_diffs[e][ref_idx]) if fcs_emotions_diffs[e] else 0,
            reverse=True,
        )[:5]
        if top5 and any(any(v != 0 for v in fcs_emotions_diffs[e]) for e in top5):
            fig, ax = plt.subplots(figsize=(12, 6))
            for emotion in top5:
                ax.plot(layers, fcs_emotions_diffs[emotion],
                        marker="o", label=emotion, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"{focus_label} − {baseline_label}")
            ax.set_title(f"[{model_short}] BBQ-{cond_name}: Layer-wise Top 5 Emotions")
            ax.legend()
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
            plt.tight_layout()
            save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig4", "layerwise")

        # ---- Predictive analysis (focus_label vs baseline_label) ----
        X_rows, y = [], []
        for r in subset:
            proj = r["emotion_projections_at_question"].get(analysis_layer, {})
            if len(proj) == len(emotions):
                if r["response_type"] == focus_label:
                    X_rows.append([proj[e] for e in sorted(emotions)])
                    y.append(1)
                elif r["response_type"] == baseline_label:
                    X_rows.append([proj[e] for e in sorted(emotions)])
                    y.append(0)
        X = np.array(X_rows)
        y = np.array(y)

        # Lowered threshold: generate AUC plot even if only 3 examples exist
        if len(X) >= 3 and len(set(y)) == 2:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            
            # Dynamic cross-validation based on smallest class size
            n_class0 = sum(y == 0)
            n_class1 = sum(y == 1)
            cv_folds = min(5, n_class0, n_class1)
            
            if cv_folds >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")
                auc_mean = float(cv_scores.mean())
                auc_std = float(cv_scores.std())
                title_str = f"(AUC={auc_mean:.3f} ± {auc_std:.3f})"
            else:
                # If there's only 1 example in a class, CV fails, but we can still fit and get importance
                auc_mean = float('nan')
                auc_std = float('nan')
                title_str = "(AUC=N/A: Needs >= 2 per class for CV)"
                
            clf.fit(X, y)
            coefs = clf.coef_[0]
            importance = pd.Series(coefs, index=sorted(emotions)).sort_values()

            fig, ax = plt.subplots(figsize=(10, 10))
            colors = ["#d32f2f" if v > 0 else "#1976d2" for v in importance.values]
            importance.plot(kind="barh", ax=ax, color=colors)
            ax.set_title(f"[{model_short}] BBQ-{cond_name} Predictive\n"
                         f"{title_str}\n"
                         f"+coef ⇒ predicts {focus_label}")
            ax.axvline(x=0, color="black", linewidth=0.8)
            plt.tight_layout()
            save_fig(output_dir, model_short, f"bbq_{cond_name}", "fig9", "predictive")

            pred_rows = [{
                "model": model_short, "context": cond_name,
                "analysis": "predictive",
                "auc_mean": auc_mean, "auc_std": auc_std,
                "n_samples": int(len(y)),
                "focus_class": focus_label, "baseline_class": baseline_label,
            }]
            for e, c in zip(sorted(emotions), coefs):
                pred_rows.append({
                    "model": model_short, "context": cond_name,
                    "analysis": "feature_importance",
                    "emotion": e, "coefficient": float(c),
                })
            save_stats_csv(pred_rows, output_dir, model_short, f"bbq_{cond_name}_predictive")
        else:
             print(f"  [skip predictive] BBQ-{cond_name}: insufficient class variation "
                   f"(N={len(y)}, classes={set(y) if len(y)>0 else '∅'})")

        # ---- t-tests with effect sizes ----
        ttest_rows = []
        for emotion in sorted(emotions):
            fv = df[(df["response_type"] == focus_label) & (df["emotion"] == emotion)]["activation"].values
            bv = df[(df["response_type"] == baseline_label) & (df["emotion"] == emotion)]["activation"].values
            if len(fv) >= 2 and len(bv) >= 2:  # Lowered t-test requirement to 2 per class
                t, p = stats.ttest_ind(fv, bv, equal_var=False)
                pooled = np.sqrt((np.var(fv) + np.var(bv)) / 2)
                d = (np.mean(fv) - np.mean(bv)) / pooled if pooled > 0 else 0
                ttest_rows.append({
                    "model": model_short, "context": cond_name, "emotion": emotion,
                    "focus_class": focus_label, "baseline_class": baseline_label,
                    "focus_mean": float(np.mean(fv)),
                    "baseline_mean": float(np.mean(bv)),
                    "diff": float(np.mean(fv) - np.mean(bv)),
                    "cohen_d": float(d),
                    "t_stat": float(t), "p_value": float(p),
                    "significance": _sig(p),
                    "n_focus": int(len(fv)), "n_baseline": int(len(bv)),
                })
        save_stats_csv(ttest_rows, output_dir, model_short, f"bbq_{cond_name}_ttests")


# ============================================================
# BOLD Analysis
# ============================================================

def analyze_bold(data, output_dir, model_short):
    results = data["results"]
    emotions = data["emotions"]
    layers = data["target_layers"]
    analysis_layer = layers[len(layers) * 2 // 3]

    print(f"\n[{model_short}] BOLD ANALYSIS (layer {analysis_layer})")

    rows = []
    for r in results:
        proj = r["emotion_projections_after_generation"].get(analysis_layer, {})
        prompt_proj = r["emotion_projections_at_prompt"].get(analysis_layer, {})
        for emotion in emotions:
            if emotion in proj and emotion in prompt_proj:
                rows.append({
                    "domain": r["domain"], "category": r["category"], "emotion": emotion,
                    "activation_after": proj[emotion],
                    "activation_before": prompt_proj[emotion],
                    "activation_change": proj[emotion] - prompt_proj[emotion],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # ---- fig7: by domain ----
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df.groupby(["emotion", "domain"])["activation_after"].mean().reset_index()
    pivot_wide = pivot.pivot(index="emotion", columns="domain", values="activation_after")
    pivot_wide.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(f"[{model_short}] BOLD: Mean Emotion Activation After Generation by Domain "
                 f"(Layer {analysis_layer})")
    ax.set_ylabel("Mean Projection")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig(output_dir, model_short, "bold", "fig7", "by_domain")

    # ---- fig8: emotion change heatmap ----
    fig, ax = plt.subplots(figsize=(14, 8))
    change_by_domain = df.groupby(["emotion", "domain"])["activation_change"].mean().reset_index()
    pivot_change = change_by_domain.pivot(index="emotion", columns="domain", values="activation_change")
    sns.heatmap(pivot_change, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".3f",
                linewidths=0.5, annot_kws={"fontsize": 7},
                cbar_kws={"label": "Activation Change (After − Before)"})
    ax.set_title(f"[{model_short}] BOLD: Emotion Activation Change During Generation")
    plt.tight_layout()
    save_fig(output_dir, model_short, "bold", "fig8", "emotion_change")

    # ---- per-domain stats CSV ----
    stat_rows = []
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        for emotion in sorted(emotions):
            es = sub[sub["emotion"] == emotion]
            if len(es) > 0:
                stat_rows.append({
                    "model": model_short, "dataset": "bold",
                    "domain": domain, "layer": analysis_layer, "emotion": emotion,
                    "mean_after_generation": float(es["activation_after"].mean()),
                    "mean_before_generation": float(es["activation_before"].mean()),
                    "mean_change": float(es["activation_change"].mean()),
                    "n": int(len(es)),
                })
    save_stats_csv(stat_rows, output_dir, model_short, "bold_domain_stats")

    # ---- layer-wise top variable emotions ----
    domain_variance = {}
    for emotion in emotions:
        means = df[df["emotion"] == emotion].groupby("domain")["activation_after"].mean()
        domain_variance[emotion] = means.var() if len(means) > 1 else 0
    top_var = sorted(domain_variance.keys(), key=lambda e: -domain_variance[e])[:5]

    if top_var:
        fig, ax = plt.subplots(figsize=(12, 6))
        for emotion in top_var:
            layer_means = []
            for layer in layers:
                vals = []
                for r in results:
                    proj = r["emotion_projections_after_generation"].get(layer, {})
                    if emotion in proj:
                        vals.append(proj[emotion])
                layer_means.append(np.mean(vals) if vals else 0)
            ax.plot(layers, layer_means, marker="o", label=emotion, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Emotion Activation")
        ax.set_title(f"[{model_short}] BOLD: Layer-wise Top 5 Domain-Variable Emotions")
        ax.legend()
        plt.tight_layout()
        save_fig(output_dir, model_short, "bold", "fig4", "layerwise")

    # ---- category heatmap (top 8) ----
    top_cats = df["category"].value_counts().head(8).index.tolist()
    if top_cats:
        cat_df = df[df["category"].isin(top_cats)]
        pivot = cat_df.groupby(["emotion", "category"])["activation_after"].mean().reset_index()
        pivot_wide = pivot.pivot(index="emotion", columns="category", values="activation_after")
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot_wide, center=0, cmap="RdBu_r", ax=ax, annot=False,
                    cbar_kws={"label": "Mean Activation"})
        ax.set_title(f"[{model_short}] BOLD: Emotion Activation by Top 8 Categories")
        plt.tight_layout()
        save_fig(output_dir, model_short, "bold", "fig6", "category")

    # ---- ANOVA across domains ----
    anova_rows = []
    for emotion in sorted(emotions):
        groups, names = [], []
        for domain in df["domain"].unique():
            vals = df[(df["domain"] == domain) & (df["emotion"] == emotion)]["activation_after"].values
            if len(vals) > 5:
                groups.append(vals); names.append(domain)
        if len(groups) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                anova_rows.append({
                    "model": model_short, "dataset": "bold", "emotion": emotion,
                    "n_domains": len(groups),
                    "f_stat": float(f_stat), "p_value": float(p_val),
                    "significance": _sig(p_val),
                    "max_mean": float(max(np.mean(g) for g in groups)),
                    "min_mean": float(min(np.mean(g) for g in groups)),
                })
            except Exception:
                pass
    save_stats_csv(anova_rows, output_dir, model_short, "bold_anova")


# ============================================================
# Cross-Dataset Comparison
# ============================================================

def analyze_cross_dataset(stereoset_data, genassoc_data, bbq_data, bold_data,
                          output_dir, model_short):
    print(f"\n[{model_short}] CROSS-DATASET ANALYSIS")

    dataset_diffs = {}

    def _three_cond_diff(data, label_in_plot):
        results = data["results"]
        emotions = data["emotions"]
        layers = data["target_layers"]
        analysis_layer = layers[len(layers) * 2 // 3]
        out = {}
        for emotion in emotions:
            sv, av = [], []
            for r in results:
                s_proj = r["emotion_projections"]["stereotype"].get(analysis_layer, {})
                a_proj = r["emotion_projections"]["anti_stereotype"].get(analysis_layer, {})
                if emotion in s_proj and emotion in a_proj:
                    sv.append(s_proj[emotion]); av.append(a_proj[emotion])
            if sv:
                out[emotion] = np.mean(sv) - np.mean(av)
        dataset_diffs[label_in_plot] = out

    if stereoset_data:
        _three_cond_diff(stereoset_data, "StereoSet\n(stereo − anti)")
    if genassoc_data:
        _three_cond_diff(genassoc_data, "GenAssocBias\n(stereo − anti)")

    if bbq_data:
        results = bbq_data["results"]
        emotions = bbq_data["emotions"]
        layers = bbq_data["target_layers"]
        analysis_layer = layers[len(layers) * 2 // 3]
        # Use ambig only: stereotyped_guess − correct
        out = {}
        for emotion in emotions:
            sg, co = [], []
            for r in results:
                # Use updated condition check for cross dataset
                if r.get("condition") != "ambig" and r.get("is_ambiguous") is not True:
                    continue
                proj = r["emotion_projections_at_question"].get(analysis_layer, {})
                if emotion in proj:
                    if r["response_type"] == "stereotyped_guess":
                        sg.append(proj[emotion])
                    elif r["response_type"] == "correct":
                        co.append(proj[emotion])
            if sg and co:
                out[emotion] = np.mean(sg) - np.mean(co)
        if out:
            dataset_diffs["BBQ-ambig\n(stereo guess − correct)"] = out

    if bold_data:
        results = bold_data["results"]
        emotions = bold_data["emotions"]
        layers = bold_data["target_layers"]
        analysis_layer = layers[len(layers) * 2 // 3]
        out = {}
        for emotion in emotions:
            rv, pv = [], []
            for r in results:
                proj = r["emotion_projections_after_generation"].get(analysis_layer, {})
                if emotion in proj:
                    if r["domain"] == "race":
                        rv.append(proj[emotion])
                    elif r["domain"] == "profession":
                        pv.append(proj[emotion])
            if rv and pv:
                out[emotion] = np.mean(rv) - np.mean(pv)
        if out:
            dataset_diffs["BOLD\n(race − profession)"] = out

    if not dataset_diffs:
        print("  No datasets available.")
        return

    all_emotions = sorted({e for d in dataset_diffs.values() for e in d.keys()})
    df = pd.DataFrame(index=all_emotions)
    for ds_name, diffs in dataset_diffs.items():
        df[ds_name] = pd.Series(diffs)
    df = df.fillna(0)
    df = df.assign(_mean=df.mean(axis=1)).sort_values("_mean").drop(columns=["_mean"])

    # ---- fig10: grouped bars ----
    fig, ax = plt.subplots(figsize=(14, 14))
    n_emotions = len(df)
    n_datasets = len(df.columns)
    bar_height = 0.8 / max(1, n_datasets)
    y_positions = np.arange(n_emotions)
    palette = ["#d32f2f", "#1976d2", "#388e3c", "#f57c00", "#7b1fa2"]
    for i, col in enumerate(df.columns):
        offset = (i - n_datasets / 2 + 0.5) * bar_height
        ax.barh(y_positions + offset, df[col], height=bar_height,
                label=col, color=palette[i % len(palette)], alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlabel("Mean Activation Difference")
    ax.set_title(f"[{model_short}] Emotion Activation Difference Across All Datasets")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    save_fig(output_dir, model_short, "cross", "fig10", "comparison")

    # ---- fig11: heatmap ----
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(df, center=0, cmap="RdBu_r", ax=ax, annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Activation Difference"},
                annot_kws={"fontsize": 8})
    ax.set_title(f"[{model_short}] Cross-Dataset Heatmap")
    ax.set_ylabel("Emotion")
    plt.tight_layout()
    save_fig(output_dir, model_short, "cross", "fig11", "heatmap")

    # ---- CSVs ----
    csv_rows = []
    for emotion in df.index:
        row = {"model": model_short, "emotion": emotion}
        for col in df.columns:
            row[col.replace("\n", " ").replace("−", "minus")] = float(df.loc[emotion, col])
        csv_rows.append(row)
    save_stats_csv(csv_rows, output_dir, model_short, "cross_dataset_diffs")

    corr_rows = []
    cols = df.columns.tolist()
    print("\n  Cross-dataset correlations:")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            valid = df[[cols[i], cols[j]]].dropna()
            if len(valid) > 3:
                r, p = stats.pearsonr(valid[cols[i]], valid[cols[j]])
                ni = cols[i].replace("\n", " "); nj = cols[j].replace("\n", " ")
                print(f"    {ni} <-> {nj}: r={r:.3f}, p={p:.4f}")
                corr_rows.append({
                    "model": model_short, "dataset_1": ni, "dataset_2": nj,
                    "pearson_r": float(r), "p_value": float(p),
                    "n_emotions": int(len(valid)),
                })
    save_stats_csv(corr_rows, output_dir, model_short, "cross_dataset_correlations")


# ============================================================
# Main
# ============================================================

def main():
    global ALL_STATS
    ALL_STATS = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_slug", type=str, required=True,
                        help="Long HF-style slug, e.g. google_gemma_2_2b_it")
    parser.add_argument("--model_short", type=str, default="",
                        help="Short identifier for filenames; auto-derived if omitted")
    parser.add_argument("--stereoset", type=str, required=True)
    parser.add_argument("--genassocbias", type=str, default="")
    parser.add_argument("--bbq", type=str, required=True)
    parser.add_argument("--bold", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/figures/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    model_short = args.model_short.strip() or shorten_slug(args.model_slug)
    print(f"Model slug : {args.model_slug}")
    print(f"Model short: {model_short}")

    stereoset_data = bbq_data = bold_data = genassoc_data = None

    if os.path.exists(args.stereoset):
        stereoset_data = load_results(args.stereoset)
        analyze_stereoset(stereoset_data, args.output, model_short)

    if args.genassocbias and os.path.exists(args.genassocbias):
        genassoc_data = load_results(args.genassocbias)
        analyze_genassocbias(genassoc_data, args.output, model_short)

    if os.path.exists(args.bbq):
        bbq_data = load_results(args.bbq)
        analyze_bbq_enhanced(bbq_data, args.output, model_short)

    if os.path.exists(args.bold):
        bold_data = load_results(args.bold)
        analyze_bold(bold_data, args.output, model_short)

    analyze_cross_dataset(stereoset_data, genassoc_data, bbq_data, bold_data,
                          args.output, model_short)

    if ALL_STATS:
        unified_path = os.path.join(args.output, f"{model_short}_ALL_STATS.csv")
        pd.DataFrame(ALL_STATS).to_csv(unified_path, index=False)
        print(f"\n{'='*70}\nUNIFIED STATS FILE: {unified_path}\n"
              f"  Total rows: {len(ALL_STATS)}\n{'='*70}")

    print(f"\n[{model_short}] All figures & CSVs saved to {args.output}")


if __name__ == "__main__":
    main()