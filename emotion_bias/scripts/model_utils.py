"""
model_utils.py - Shared utilities for model loading and activation extraction.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple


def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer with appropriate settings."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",  # needed for reliable hidden state extraction
    )
    model.eval()
    print(f"Model loaded. Num layers: {model.config.num_hidden_layers}, "
          f"Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


def get_hidden_states(
    model,
    tokenizer,
    text: str,
    layers: Optional[List[int]] = None,
    token_position: str = "last",  # "last" or "all"
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states (residual stream) at specified layers.

    Returns:
        Dict mapping layer_index -> tensor of shape (hidden_dim,) for last token,
        or (seq_len, hidden_dim) for all tokens.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
    num_layers = len(hidden_states) - 1  # exclude embedding layer

    if layers is None:
        layers = list(range(num_layers))

    result = {}
    for layer_idx in layers:
        hs = hidden_states[layer_idx + 1][0]  # +1 to skip embedding layer, [0] for batch
        if token_position == "last":
            # Get the last non-padding token
            seq_len = inputs["attention_mask"].sum().item()
            result[layer_idx] = hs[seq_len - 1].float().cpu()
        else:
            result[layer_idx] = hs.float().cpu()

    return result


def get_hidden_states_batch(
    model,
    tokenizer,
    texts: List[str],
    layers: Optional[List[int]] = None,
    token_position: str = "last",
    batch_size: int = 8,
) -> List[Dict[int, torch.Tensor]]:
    """Batch version of get_hidden_states for efficiency."""
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1

        if layers is None:
            layers = list(range(num_layers))

        for b in range(len(batch_texts)):
            result = {}
            seq_len = inputs["attention_mask"][b].sum().item()
            for layer_idx in layers:
                hs = hidden_states[layer_idx + 1][b]
                if token_position == "last":
                    result[layer_idx] = hs[seq_len - 1].float().cpu()
                else:
                    result[layer_idx] = hs[:seq_len].float().cpu()
            all_results.append(result)

    return all_results


def compute_emotion_projections(
    hidden_state: torch.Tensor,
    emotion_vectors: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Project a hidden state onto each emotion vector direction.

    Args:
        hidden_state: (hidden_dim,) tensor
        emotion_vectors: dict mapping emotion_name -> (hidden_dim,) unit vector

    Returns:
        Dict mapping emotion_name -> projection score (dot product)
    """
    projections = {}
    for emotion_name, emotion_vec in emotion_vectors.items():
        # Cosine similarity (emotion vectors are already normalized)
        proj = torch.dot(hidden_state, emotion_vec).item()
        projections[emotion_name] = proj
    return projections


def compute_emotion_projections_batch(
    hidden_states: List[torch.Tensor],
    emotion_vectors: Dict[str, torch.Tensor],
) -> List[Dict[str, float]]:
    """Batch version of compute_emotion_projections."""
    # Stack hidden states: (N, hidden_dim)
    H = torch.stack(hidden_states)
    # Stack emotion vectors: (num_emotions, hidden_dim)
    emotion_names = list(emotion_vectors.keys())
    E = torch.stack([emotion_vectors[name] for name in emotion_names])
    # Compute all projections at once: (N, num_emotions)
    projections = H @ E.T
    # Convert to list of dicts
    results = []
    for i in range(len(hidden_states)):
        result = {name: projections[i, j].item() for j, name in enumerate(emotion_names)}
        results.append(result)
    return results


def get_next_token_probs(
    model, tokenizer, text: str, top_k: int = 10
) -> List[Tuple[str, float]]:
    """Get top-k next token probabilities."""
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [(tokenizer.decode(tid), p.item()) for tid, p in zip(top_ids, top_probs)]


def get_completion_log_prob(
    model, tokenizer, context: str, completion: str
) -> float:
    """Get log probability of a completion given a context."""
    full_text = context + completion
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
    context_len = context_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # (seq_len, vocab_size)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Sum log probs of completion tokens
    total_log_prob = 0.0
    input_ids = inputs["input_ids"][0]
    for i in range(context_len, len(input_ids)):
        token_id = input_ids[i]
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob
