"""
probe_bold.py - Fixed list flattening and silent skipping.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model_utils import (
    load_model, get_hidden_states, compute_emotion_projections
)

def generate_and_probe(
    model, tokenizer, prompt: str,
    emotion_vectors_by_layer: dict, target_layers: list,
    max_new_tokens: int = 50,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    emotion_projections_before = {}
    emotion_projections_after = {}

    # Prompt-only projections
    hs_before = get_hidden_states(model, tokenizer, prompt, layers=target_layers)
    for l in target_layers:
        emotion_projections_before[l] = compute_emotion_projections(hs_before[l], emotion_vectors_by_layer[l])

    # Final text projections
    full_text = prompt + generated_text
    hs_after = get_hidden_states(model, tokenizer, full_text, layers=target_layers)
    for l in target_layers:
        emotion_projections_after[l] = compute_emotion_projections(hs_after[l], emotion_vectors_by_layer[l])

    return {
        "generated_text": generated_text,
        "emotion_projections_at_prompt": emotion_projections_before,
        "emotion_projections_after_generation": emotion_projections_after,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--emotion_vectors", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--layers", type=str, default="quarter")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ev_data = torch.load(args.emotion_vectors, weights_only=False)
    emotion_vectors_by_layer = ev_data["vectors"]
    num_layers = ev_data["num_layers"]
    
    all_layers = sorted(emotion_vectors_by_layer.keys())
    if args.layers == "quarter":
        target_layers = [l for l in all_layers if l % (num_layers // 6) == 0 or l == num_layers - 1]
    else:
        target_layers = [int(x) for x in args.layers.split(",")]

    model, tokenizer = load_model(args.model)

    with open(args.data, 'r') as f:
        bold_data = json.load(f)

    # --- ROBUST DATA PARSING ---
    all_prompts = []
    
    if isinstance(bold_data, dict):
        # Standard dict structure: {domain: {category: {name: [prompts]}}}
        for domain, cats in bold_data.items():
            for cat, names in cats.items():
                for name, prompts in names.items():
                    if isinstance(prompts, list):
                        for p in prompts:
                            all_prompts.append({
                                "domain": domain, "category": cat, "name": name, "prompt": p
                            })
                    else:
                        all_prompts.append({
                            "domain": domain, "category": cat, "name": name, "prompt": prompts
                        })
                        
    elif isinstance(bold_data, list):
        # Flatten list structure (e.g., from HF datasets export)
        for item in bold_data:
            # Check for prompts (list) or prompt/text (string)
            p_val = item.get("prompts") or item.get("prompt") or item.get("text")
            
            domain = item.get("domain", "unknown")
            cat = item.get("category", "unknown")
            name = item.get("name", "unknown")
            
            if isinstance(p_val, list):
                for p in p_val:
                    all_prompts.append({
                        "domain": domain, "category": cat, "name": name, "prompt": p
                    })
            elif isinstance(p_val, str):
                all_prompts.append({
                    "domain": domain, "category": cat, "name": name, "prompt": p_val
                })
    else:
        print(f"[Error] Unexpected BOLD data type: {type(bold_data)}")
        return

    if not all_prompts:
        print("[Error] No prompts could be extracted. Check JSON keys (expects 'prompts', 'prompt', or 'text').")
        return

    # Apply sampling
    if args.max_examples > 0 and len(all_prompts) > args.max_examples:
        np.random.seed(42)
        indices = np.random.choice(len(all_prompts), args.max_examples, replace=False)
        all_prompts = [all_prompts[i] for i in indices]

    results = []
    for item in tqdm(all_prompts, desc="Probing BOLD"):
        try:
            # p_text is now guaranteed to exist and be a string
            p_text = item["prompt"] 
            
            res = generate_and_probe(
                model, tokenizer, p_text, 
                emotion_vectors_by_layer, target_layers, 
                args.max_new_tokens
            )
            # Merge original metadata with new probing results
            results.append({**item, **res})
        except Exception as e:
            # Expose the error so it doesn't fail silently
            if len(results) == 0:
                print(f"\n[Debug] BOLD Error on generation: {e}")
            continue

    # Save output
    torch.save({
        "dataset": "bold",
        "model": args.model,
        "target_layers": target_layers,
        "emotions": ev_data["emotions"],
        "results": results,
    }, args.output)
    
    print(f"\nBOLD Summary: Probed {len(results)} examples. Results saved to {args.output}")

if __name__ == "__main__":
    main()