"""
mmlu_eval.py -- Evaluate MMLU accuracy on a model, optionally under a
steering hook.

Loads a stratified sub-sample of MMLU (test split, one question per batch),
formats each as a 4-way multiple choice prompt, and scores by taking the
argmax of P(" A" | prompt), P(" B" | prompt), P(" C" | prompt), P(" D" | prompt)
at the last position.

When steering flags are provided, wraps the forward pass in the same
SteeringHook used by probe_with_steering.py, so capability is evaluated
under EXACTLY the same intervention that was applied to the bias probes.

Usage (baseline -- no steering):
    python scripts/mmlu_eval.py \
        --model google/gemma-2-2b-it \
        --output outputs/<slug>_mmlu_baseline.pt \
        --n_questions 500

Usage (steered -- matches a specific run of probe_with_steering.py):
    python scripts/mmlu_eval.py \
        --model google/gemma-2-2b-it \
        --output outputs/<slug>_mmlu_alpha25.pt \
        --n_questions 500 \
        --emotion_vectors outputs/<slug>_emotion_vectors.pt \
        --mitigation_directions outputs/<slug>_mitigation_directions.pt \
        --scope pooled --method logreg --subspace task \
        --alpha 25 --target_layers 20
"""

import os
import sys
import argparse
from typing import List

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_utils import load_model  # noqa: E402
from mitigation_utils import build_V_matrix  # noqa: E402
from probe_with_steering import SteeringHook  # noqa: E402


# ============================================================
# MMLU loading (stratified subsample)
# ============================================================

def load_mmlu_subsample(n_questions: int = 500, seed: int = 42) -> List[dict]:
    """
    Load cais/mmlu (all subjects), test split, and stratify-sample n_questions
    so every subject is represented roughly equally.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    last_err = None
    for path in ["cais/mmlu", "hendrycks_test"]:
        try:
            ds = load_dataset(path, "all", split="test")
            print(f"[mmlu] Loaded {len(ds)} questions from '{path}' (all/test)")
            break
        except Exception as e:
            last_err = e
            print(f"[mmlu] Failed to load '{path}': {type(e).__name__}: {e}")
    else:
        raise RuntimeError(f"Could not load MMLU. Last error: {last_err}")

    import pandas as pd
    df = pd.DataFrame({
        "question": ds["question"],
        "subject": ds["subject"],
        "choices": ds["choices"],
        "answer": ds["answer"],
    })
    subjects = sorted(df["subject"].unique().tolist())
    per_subj = max(1, n_questions // len(subjects))
    sampled = (
        df.groupby("subject", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), per_subj), random_state=seed))
          .reset_index(drop=True)
    )
    # Top up to exactly n_questions if stratification undershot
    if len(sampled) < n_questions:
        remaining = df[~df.index.isin(sampled.index)]
        extra_n = min(n_questions - len(sampled), len(remaining))
        if extra_n > 0:
            extra = remaining.sample(n=extra_n, random_state=seed)
            sampled = pd.concat([sampled, extra]).reset_index(drop=True)
    if len(sampled) > n_questions:
        sampled = sampled.sample(n=n_questions, random_state=seed).reset_index(drop=True)
    print(f"[mmlu] Stratified subsample: {len(sampled)} questions across "
          f"{sampled['subject'].nunique()} subjects")
    return sampled.to_dict(orient="records")


# ============================================================
# Prompt formatting + letter-token discovery
# ============================================================

def format_prompt(q: dict) -> str:
    """Standard MMLU prompt: Question + A. B. C. D. + Answer:"""
    s = f"Question: {q['question']}\n"
    for i, ch in enumerate(q["choices"]):
        s += f"{chr(ord('A') + i)}. {ch}\n"
    s += "Answer:"
    return s


def get_letter_token_ids(tokenizer) -> List[int]:
    """
    Token IDs for ' A', ' B', ' C', ' D' (with leading space, which is how
    these letters appear after 'Answer:' in the prompt). Robust to whether
    the tokenizer treats them as single tokens or not.
    """
    ids = []
    for letter in "ABCD":
        toks = tokenizer(" " + letter, add_special_tokens=False).input_ids
        # Take the last token; this is the letter itself for both single-token
        # and multi-token cases (SentencePiece tokenizers sometimes prepend a
        # space token; we want the letter, not the space).
        ids.append(toks[-1])
    return ids


# ============================================================
# Scoring loop
# ============================================================

@torch.no_grad()
def score_questions(
    model,
    tokenizer,
    questions: List[dict],
    batch_size: int = 8,
    max_length: int = 1024,
):
    device = next(model.parameters()).device
    letter_ids = get_letter_token_ids(tokenizer)

    # Left-pad so that the final real token of every sequence lives at
    # position -1, regardless of prompt length.
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    preds, correct = [], 0
    try:
        for start in tqdm(range(0, len(questions), batch_size), desc="[mmlu] scoring"):
            batch = questions[start:start + batch_size]
            prompts = [format_prompt(q) for q in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            out = model(**enc)
            # With left-padding, index -1 is always the last real token.
            last_logits = out.logits[:, -1, :]           # (B, V)
            letter_logits = last_logits[:, letter_ids]   # (B, 4)
            batch_preds = letter_logits.argmax(dim=-1).cpu().numpy()
            for p, q in zip(batch_preds, batch):
                preds.append(int(p))
                if int(p) == int(q["answer"]):
                    correct += 1
    finally:
        tokenizer.padding_side = prev_padding_side

    accuracy = correct / max(1, len(questions))
    return preds, accuracy


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n_questions", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # Steering (all-or-nothing group; if any of the core 4 are missing, run
    # baseline)
    ap.add_argument("--emotion_vectors", type=str, default=None)
    ap.add_argument("--mitigation_directions", type=str, default=None)
    ap.add_argument("--scope", type=str, default=None,
                    choices=["stereoset", "genassoc", "bbq_ambig", "pooled"])
    ap.add_argument("--method", type=str, default=None,
                    choices=["diff", "lda", "pls", "logreg"])
    ap.add_argument("--subspace", type=str, default=None,
                    choices=["variance", "task"])
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--target_layers", type=str, default=None)
    ap.add_argument("--conditional", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    steering_on = all([
        args.emotion_vectors, args.mitigation_directions,
        args.scope, args.method, args.subspace,
        args.alpha is not None, args.target_layers,
    ])

    questions = load_mmlu_subsample(args.n_questions, seed=args.seed)
    model, tokenizer = load_model(args.model)

    if steering_on:
        ev = torch.load(args.emotion_vectors, weights_only=False)
        emotions_order = ev["emotions"]
        vectors_by_layer = ev["vectors"]
        mit = torch.load(args.mitigation_directions, weights_only=False)
        scope_data = mit["scopes"][args.scope]

        target_layers = [int(x) for x in args.target_layers.split(",")]
        for l in target_layers:
            if l not in scope_data:
                raise ValueError(
                    f"No mitigation direction for scope='{args.scope}' at layer {l}. "
                    f"Available: {sorted(scope_data.keys())}"
                )

        sub_key = f"{args.subspace}_subspace"
        d_perp_per_layer, V_per_layer, w_per_layer, tau_per_layer = {}, {}, {}, {}
        for l in target_layers:
            m = scope_data[l]["methods"][args.method]
            d_perp_per_layer[l] = m[sub_key]["d_perp"]
            V_per_layer[l] = build_V_matrix(vectors_by_layer[l], emotions_order)
            w_per_layer[l] = m["w"]
            tau_per_layer[l] = args.tau if args.tau is not None else m["tau_suggest"]

        print(f"[mmlu] Steering ON: scope={args.scope}, method={args.method}, "
              f"subspace={args.subspace}, alpha={args.alpha}, layers={target_layers}, "
              f"conditional={args.conditional}")
        for l in target_layers:
            norm_perp = scope_data[l]["methods"][args.method][sub_key]["norm_perp"]
            print(f"    layer {l}: ||d_perp||={norm_perp:.3f}  "
                  f"alpha*||d_perp||={args.alpha * norm_perp:.3f}")

        with SteeringHook(
            model=model, target_layers=target_layers,
            d_perp_per_layer=d_perp_per_layer, alpha=args.alpha,
            V_per_layer=V_per_layer, w_per_layer=w_per_layer,
            tau_per_layer=tau_per_layer, conditional=args.conditional,
        ):
            preds, accuracy = score_questions(
                model, tokenizer, questions, batch_size=args.batch_size,
            )
    else:
        print("[mmlu] Running WITHOUT steering (baseline)")
        preds, accuracy = score_questions(
            model, tokenizer, questions, batch_size=args.batch_size,
        )

    # Per-subject accuracy
    per_subj: dict = {}
    for q, p in zip(questions, preds):
        per_subj.setdefault(q["subject"], []).append(int(int(p) == int(q["answer"])))
    subj_accs = {s: float(np.mean(v)) for s, v in per_subj.items()}

    print(f"\n[mmlu] Overall accuracy: {accuracy:.4f} "
          f"({int(accuracy * len(questions))}/{len(questions)})")
    hardest = sorted(subj_accs.items(), key=lambda kv: kv[1])[:5]
    easiest = sorted(subj_accs.items(), key=lambda kv: -kv[1])[:5]
    print(f"[mmlu] Hardest subjects: " + ", ".join(f"{s}={a:.2f}" for s, a in hardest))
    print(f"[mmlu] Easiest subjects: " + ", ".join(f"{s}={a:.2f}" for s, a in easiest))

    payload = {
        "model": args.model,
        "n_questions": len(questions),
        "accuracy": float(accuracy),
        "preds": preds,
        "answers": [int(q["answer"]) for q in questions],
        "subjects": [q["subject"] for q in questions],
        "subject_accuracy": subj_accs,
        "steering": None if not steering_on else {
            "scope": args.scope, "method": args.method, "subspace": args.subspace,
            "alpha": args.alpha, "target_layers": target_layers,
            "conditional": args.conditional,
        },
    }
    torch.save(payload, args.output)
    print(f"[mmlu] Saved results to: {args.output}")


if __name__ == "__main__":
    main()
