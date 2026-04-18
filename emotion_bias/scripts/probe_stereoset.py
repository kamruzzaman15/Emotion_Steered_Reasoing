"""
probe_stereoset.py - Probe emotion vector activations on StereoSet.

For each StereoSet example, we measure:
1. Log-probability of stereotype vs anti-stereotype completion
2. Emotion vector activations when the model processes each completion
3. Difference in emotion profiles between stereotype and anti-stereotype
"""

import os
import json
import argparse
import torch
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
    """Probe emotion activations for a single StereoSet example."""
    context = example["context"]
    stereo = example["stereotype"]
    anti_stereo = example["anti_stereotype"]
    unrelated = example["unrelated"]

    # Compute log-probs for model preference
    lp_stereo = get_completion_log_prob(model, tokenizer, context + " ", stereo)
    lp_anti = get_completion_log_prob(model, tokenizer, context + " ", anti_stereo)
    lp_unrelated = get_completion_log_prob(model, tokenizer, context + " ", unrelated)

    prefers_stereotype = lp_stereo > lp_anti

    # Extract hidden states for each completion
    result = {
        "id": example["id"],
        "bias_type": example["bias_type"],
        "target": example["target"],
        "prefers_stereotype": prefers_stereotype,
        "log_prob_stereo": lp_stereo,
        "log_prob_anti": lp_anti,
        "log_prob_unrelated": lp_unrelated,
        "emotion_projections": {},
    }

    for condition_name, text in [
        ("stereotype", context + " " + stereo),
        ("anti_stereotype", context + " " + anti_stereo),
        ("unrelated", context + " " + unrelated),
    ]:
        hs = get_hidden_states(model, tokenizer, text, layers=target_layers)
        layer_projections = {}
        for layer_idx in target_layers:
            emotion_vecs = emotion_vectors_by_layer[layer_idx]
            projections = compute_emotion_projections(hs[layer_idx], emotion_vecs)
            layer_projections[layer_idx] = projections
        result["emotion_projections"][condition_name] = layer_projections

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--emotion_vectors", type=str, default="outputs/emotion_vectors.pt")
    parser.add_argument("--data", type=str, default="data/stereoset.json")
    parser.add_argument("--output", type=str, default="outputs/stereoset_results.pt")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples for debugging")
    parser.add_argument("--layers", type=str, default="quarter",
                        help="'all', 'quarter' (every 4th), or comma-separated indices")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load emotion vectors
    ev_data = torch.load(args.emotion_vectors, weights_only=False)
    emotion_vectors_by_layer = ev_data["vectors"]
    all_layers = sorted(emotion_vectors_by_layer.keys())
    num_layers = ev_data["num_layers"]

    # Determine target layers
    if args.layers == "all":
        target_layers = all_layers
    elif args.layers == "quarter":
        target_layers = [l for l in all_layers if l % (num_layers // 6) == 0 or l == num_layers - 1]
    else:
        target_layers = [int(x) for x in args.layers.split(",")]
    target_layers = [l for l in target_layers if l in emotion_vectors_by_layer]
    print(f"Probing layers: {target_layers}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Load dataset
    with open(args.data) as f:
        dataset = json.load(f)
    if args.max_examples:
        dataset = dataset[:args.max_examples]
    print(f"Loaded {len(dataset)} StereoSet examples")

    # Probe
    results = []
    for example in tqdm(dataset, desc="Probing StereoSet"):
        try:
            result = probe_single_example(
                model, tokenizer, example,
                emotion_vectors_by_layer, target_layers
            )
            results.append(result)
        except Exception as e:
            print(f"Error on {example['id']}: {e}")
            continue

    # Save results
    torch.save({
        "dataset": "stereoset",
        "model": args.model,
        "target_layers": target_layers,
        "emotions": ev_data["emotions"],
        "results": results,
    }, args.output)

    # Print summary
    n_stereo = sum(1 for r in results if r["prefers_stereotype"])
    print(f"\nStereoSet Summary:")
    print(f"  Total examples: {len(results)}")
    print(f"  Prefers stereotype: {n_stereo} ({100*n_stereo/len(results):.1f}%)")
    print(f"  Prefers anti-stereotype: {len(results)-n_stereo} ({100*(len(results)-n_stereo)/len(results):.1f}%)")

    # Quick emotion summary at middle layer
    mid_layer = target_layers[len(target_layers) // 2]
    emotions = ev_data["emotions"]
    print(f"\n  Mean emotion projections at layer {mid_layer}:")
    for emotion in emotions[:10]:
        stereo_vals = [r["emotion_projections"]["stereotype"][mid_layer][emotion] for r in results]
        anti_vals = [r["emotion_projections"]["anti_stereotype"][mid_layer][emotion] for r in results]
        diff = np.mean(stereo_vals) - np.mean(anti_vals)
        print(f"    {emotion:20s}: stereo={np.mean(stereo_vals):+.4f}  anti={np.mean(anti_vals):+.4f}  diff={diff:+.4f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
