"""
probe_genassocbias.py - Probe emotion vector activations on GenAssocBias.
Updated with enhanced terminal summary for Stereo vs. Anti comparison.
"""

import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_utils import (
    load_model, get_hidden_states, compute_emotion_projections,
    get_completion_log_prob
)

def probe_single_example(
    model, tokenizer, example: dict,
    emotion_vectors_by_layer: dict, target_layers: list
):
    """Probe emotion activations for a single GenAssocBias example."""
    context_template = example["context"]
    stereo = str(example["stereotype"])
    anti_stereo = str(example["anti_stereotype"])
    unrelated = str(example["unrelated"])

    # Handle "BLANK" placeholder logic
    if "BLANK" in context_template:
        prefix, suffix = context_template.split("BLANK")
    else:
        prefix, suffix = context_template, ""

    # Compute log-probs for model preference
    lp_stereo = get_completion_log_prob(model, tokenizer, prefix, stereo + suffix)
    lp_anti = get_completion_log_prob(model, tokenizer, prefix, anti_stereo + suffix)
    lp_unrelated = get_completion_log_prob(model, tokenizer, prefix, unrelated + suffix)

    prefers_stereotype = lp_stereo > lp_anti

    # Extract hidden states for each completion
    emotion_projections = {}
    for condition, completion_word in [
        ("stereotype", stereo), 
        ("anti_stereotype", anti_stereo), 
        ("unrelated", unrelated)
    ]:
        full_text = prefix + completion_word + suffix
        hs = get_hidden_states(model, tokenizer, full_text, layers=target_layers)
        
        cond_projections = {}
        for layer_idx in target_layers:
            layer_hs = hs[layer_idx]
            cond_projections[layer_idx] = compute_emotion_projections(
                layer_hs, emotion_vectors_by_layer[layer_idx]
            )
        emotion_projections[condition] = cond_projections

    return {
        "id": example.get("id", "unknown"),
        "bias_type": example.get("bias_type", "unknown"),
        "prefers_stereotype": prefers_stereotype,
        "emotion_projections": emotion_projections
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--emotion_vectors", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--layers", type=str, default="quarter")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ev_data = torch.load(args.emotion_vectors, weights_only=False)
    emotion_vectors_by_layer = ev_data["vectors"]
    num_layers = ev_data["num_layers"]
    emotions = ev_data["emotions"]
    
    all_layers = sorted(emotion_vectors_by_layer.keys())
    if args.layers == "quarter":
        target_layers = [l for l in all_layers if l % (num_layers // 6) == 0 or l == num_layers - 1]
    else:
        target_layers = [int(x) for x in args.layers.split(",")]

    model, tokenizer = load_model(args.model)
    df = pd.read_csv(args.data)
    if args.max_examples > 0 and len(df) > args.max_examples:
        df = df.sample(n=args.max_examples, random_state=42)
    
    dataset = df.to_dict(orient="records")
    results = []

    for i, example in enumerate(tqdm(dataset, desc="Probing GenAssocBias")):
        example["id"] = f"genassoc_{i}"
        try:
            results.append(probe_single_example(model, tokenizer, example, emotion_vectors_by_layer, target_layers))
        except Exception as e:
            continue

    # Save results
    torch.save({
        "dataset": "genassocbias",
        "model": args.model,
        "target_layers": target_layers,
        "emotions": emotions,
        "results": results,
    }, args.output)

    # --- HIGH-LEVEL SUMMARY OVERVIEW ---
    n_total = len(results)
    n_stereo = sum(1 for r in results if r["prefers_stereotype"])
    n_anti = n_total - n_stereo
    
    print(f"\nGenAssocBias Summary:")
    print(f"  Total examples: {n_total}")
    print(f"  Prefers stereotype: {n_stereo} ({100*n_stereo/n_total:.1f}%)")
    print(f"  Prefers anti-stereotype: {n_anti} ({100*n_anti/n_total:.1f}%)")

    # Display projections for middle layer (matching your StereoSet layer 16 logic)
    summary_layer = target_layers[len(target_layers) // 2] if target_layers else 0
    print(f"\n  Mean emotion projections at layer {summary_layer}:")
    
    # Specific emotions to display for comparative analysis
    display_emotions = [
        "confident", "certain", "assured", "decisive", "comfortable", 
        "uncertain", "doubtful", "hesitant", "conflicted", "ambivalent"
    ]
    
    for emotion in display_emotions:
        if emotion not in emotions:
            continue
        stereo_vals = [r["emotion_projections"]["stereotype"][summary_layer][emotion] for r in results]
        anti_vals = [r["emotion_projections"]["anti_stereotype"][summary_layer][emotion] for r in results]
        s_mean, a_mean = np.mean(stereo_vals), np.mean(anti_vals)
        diff = s_mean - a_mean
        print(f"    {emotion:15s} : stereo={s_mean:+8.4f}  anti={a_mean:+8.4f}  diff={diff:+8.4f}")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()