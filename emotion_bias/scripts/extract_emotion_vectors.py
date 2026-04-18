"""
extract_emotion_vectors.py - Extract emotion concept vectors from Gemma-2-2B.

Following Anthropic (2026) and Tak et al. (2025):
1. For each emotion concept, generate short stories featuring that emotion
2. Record residual stream activations at the final token position
3. Compute per-emotion mean activation
4. Subtract global mean to get emotion direction vectors
5. Normalize to unit vectors
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model_utils import load_model, get_hidden_states


STORY_GENERATION_TEMPLATE = (
    "Write a very short story (3-4 sentences) about a person who is feeling {emotion}. "
    "Show the emotion through their thoughts and actions, not by naming the emotion directly."
)

# Diverse seed scenarios to get varied stories per emotion
SEED_SCENARIOS = [
    "at work during a meeting",
    "walking home alone at night",
    "reading a letter they just received",
    "sitting in a doctor's waiting room",
    "looking at old photographs",
    "standing at a crossroads in their life",
    "after receiving unexpected news",
    "during a conversation with a stranger",
    "while making a difficult decision",
    "watching the sunset from a park bench",
    "preparing for an important interview",
    "after finishing a long project",
    "while cooking dinner for their family",
    "during a phone call with an old friend",
    "at a crowded train station",
    "while writing in their journal",
    "after losing something important",
    "during a quiet morning routine",
    "while helping someone in need",
    "at a celebration they didn't expect",
]


def generate_emotion_stories(model, tokenizer, emotion: str, num_stories: int = 20) -> list:
    """Generate short stories for a given emotion using the model itself."""
    stories = []
    for i in range(num_stories):
        scenario = SEED_SCENARIOS[i % len(SEED_SCENARIOS)]
        prompt = (
            f"Write a very short story (3-4 sentences) about a person who is "
            f"feeling {emotion} while {scenario}. Show the emotion through their "
            f"thoughts and actions.\n\nStory:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_text = prompt + " " + generated.strip()
        stories.append(full_text)

    return stories


def extract_vectors_for_layer(
    model, tokenizer, emotion_stories: dict, layer_idx: int
) -> dict:
    """Extract emotion vectors for a single layer."""
    all_activations = []
    emotion_activations = {}

    for emotion, stories in emotion_stories.items():
        acts = []
        for story in stories:
            hs = get_hidden_states(model, tokenizer, story, layers=[layer_idx])
            acts.append(hs[layer_idx])
            all_activations.append(hs[layer_idx])
        emotion_activations[emotion] = torch.stack(acts)  # (num_stories, hidden_dim)

    # Global mean across all emotions and stories
    global_mean = torch.stack(all_activations).mean(dim=0)  # (hidden_dim,)

    # Compute emotion vectors: per-emotion mean minus global mean, then normalize
    emotion_vectors = {}
    for emotion, acts in emotion_activations.items():
        emotion_mean = acts.mean(dim=0)  # (hidden_dim,)
        direction = emotion_mean - global_mean
        # Normalize to unit vector
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
        emotion_vectors[emotion] = direction

    return emotion_vectors, global_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--config", type=str, default="configs/emotion_concepts.json")
    parser.add_argument("--output", type=str, default="outputs/emotion_vectors.pt")
    parser.add_argument("--stories_output", type=str, default="outputs/emotion_stories.json")
    parser.add_argument("--num_stories", type=int, default=20)
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layer indices, or 'all'")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Flatten emotion list
    all_emotions = []
    for cluster_name, emotions in config["emotions"].items():
        all_emotions.extend(emotions)
    print(f"Extracting vectors for {len(all_emotions)} emotions: {all_emotions}")

    # Load model
    model, tokenizer = load_model(args.model)
    num_layers = model.config.num_hidden_layers

    # Determine layers
    if args.layers == "all":
        target_layers = list(range(num_layers))
    else:
        target_layers = [int(x) for x in args.layers.split(",")]

    # Step 1: Generate stories for each emotion (reuse existing if available)
    print("\n=== Step 1: Generating emotion stories ===")
    emotion_stories = {}

    # Load existing stories if available
    existing_stories = {}
    if os.path.exists(args.stories_output):
        with open(args.stories_output) as f:
            existing_stories = json.load(f)
        print(f"  Loaded existing stories for {len(existing_stories)} emotions")

    for emotion in tqdm(all_emotions, desc="Generating stories"):
        if emotion in existing_stories and len(existing_stories[emotion]) >= args.num_stories:
            emotion_stories[emotion] = existing_stories[emotion][:args.num_stories]
            print(f"  {emotion}: reusing {len(emotion_stories[emotion])} existing stories")
        else:
            stories = generate_emotion_stories(model, tokenizer, emotion, args.num_stories)
            emotion_stories[emotion] = stories
            print(f"  {emotion}: {len(stories)} NEW stories generated")

    # Save all stories (existing + new) for reproducibility
    with open(args.stories_output, "w") as f:
        json.dump(emotion_stories, f, indent=2)
    print(f"Stories saved to {args.stories_output}")

    # Step 2: Extract emotion vectors per layer
    print("\n=== Step 2: Extracting emotion vectors ===")
    all_layer_vectors = {}
    all_layer_global_means = {}

    for layer_idx in tqdm(target_layers, desc="Extracting per layer"):
        vectors, global_mean = extract_vectors_for_layer(
            model, tokenizer, emotion_stories, layer_idx
        )
        all_layer_vectors[layer_idx] = vectors
        all_layer_global_means[layer_idx] = global_mean

    # Step 3: Compute inter-emotion cosine similarities for validation
    print("\n=== Step 3: Validation - inter-emotion similarities ===")
    mid_layer = num_layers // 2
    if mid_layer in all_layer_vectors:
        vectors_mid = all_layer_vectors[mid_layer]
        print(f"\nCosine similarities at layer {mid_layer}:")
        emotions_list = list(vectors_mid.keys())
        for i in range(min(5, len(emotions_list))):
            for j in range(i + 1, min(5, len(emotions_list))):
                e1, e2 = emotions_list[i], emotions_list[j]
                cos_sim = torch.dot(vectors_mid[e1], vectors_mid[e2]).item()
                print(f"  {e1} <-> {e2}: {cos_sim:.3f}")

    # Save everything
    save_data = {
        "model_name": args.model,
        "num_layers": num_layers,
        "hidden_dim": model.config.hidden_size,
        "emotions": all_emotions,
        "target_layers": target_layers,
        "vectors": {
            layer_idx: {emotion: vec.clone() for emotion, vec in vectors.items()}
            for layer_idx, vectors in all_layer_vectors.items()
        },
        "global_means": {
            layer_idx: mean.clone()
            for layer_idx, mean in all_layer_global_means.items()
        },
    }
    torch.save(save_data, args.output)
    print(f"\nEmotion vectors saved to {args.output}")
    print(f"Shape per vector: ({model.config.hidden_size},)")
    print(f"Total vectors: {len(all_emotions)} emotions x {len(target_layers)} layers")


if __name__ == "__main__":
    main()
