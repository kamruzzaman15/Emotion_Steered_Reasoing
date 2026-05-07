#!/bin/bash
# run_mitigation.sh -- Mitigation pipeline for a single model (Gemma-2-2B by default).
#
# Pipeline:
#   1. REUSE existing emotion vectors and existing probe results (error if missing)
#   2. Build the protected subspace from GoEmotions (Stage A)
#   3. Compute per-dataset + pooled bias directions via all 4 methods (Stages B+C)
#   4. Re-run probes with steering at a single target layer (Stage D)
#   5. Analyze baseline vs steered and generate figures
#
# Ablations are easy to kick off by re-running from Step 4 with different
# --scope / --method / --subspace / --alpha / --target_layers / --conditional flags.
#
# Usage:
#   bash run_mitigation.sh                              # defaults: gemma-2-2b-it
#   bash run_mitigation.sh google/gemma-2-2b-it
#   bash run_mitigation.sh <hf_model> <alpha> <scope> <method> <subspace> <layers>

set -e
cd "$(dirname "$0")"

MODEL="${1:-google/gemma-2-2b-it}"
ALPHA="${2:-1.0}"
SCOPE="${3:-pooled}"             # stereoset | genassoc | bbq_ambig | pooled
METHOD="${4:-logreg}"            # diff | lda | pls | logreg
SUBSPACE="${5:-task}"            # variance | task
STEER_LAYERS="${6:-}"            # comma-separated, e.g. "15". If empty, auto-pick 2/3 depth layer.
CONDITIONAL="${CONDITIONAL:-0}"   # env var: 1 to turn on conditional steering

slug=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '.' '_')

EV="outputs/${slug}_emotion_vectors.pt"
SS_BASE="outputs/${slug}_stereoset_results.pt"
GA_BASE="outputs/${slug}_genassoc_results.pt"
BBQ_BASE="outputs/${slug}_bbq_results.pt"

SUBSPACE_OUT="outputs/${slug}_protected_subspace.pt"
HIDDEN_CACHE="outputs/${slug}_goemotions_hidden.pt"
DIR_OUT="outputs/${slug}_mitigation_directions.pt"

SS_STEER="outputs/${slug}_stereoset_steered.pt"
GA_STEER="outputs/${slug}_genassoc_steered.pt"
BBQ_STEER="outputs/${slug}_bbq_steered.pt"

mkdir -p outputs outputs/figures

# ---------- Sanity checks on reusable artifacts ----------
echo "=== Sanity check: emotion vectors + baseline probes ==="
for f in "$EV" "$SS_BASE" "$GA_BASE" "$BBQ_BASE"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        echo "  Run the baseline pipeline (run_all.sh) for $MODEL first."
        exit 1
    else
        echo "  OK: $f"
    fi
done

# Resolve MODEL_SHORT using the shortener already in analyze_results.py
MODEL_SHORT=$(python -c "from scripts.analyze_results import shorten_slug; print(shorten_slug('$slug'))")
echo "  MODEL_SHORT=$MODEL_SHORT"

# ---------- Step 1: Protected subspace (Stage A) ----------
if [ -f "$SUBSPACE_OUT" ]; then
    echo ""
    echo "=== Step 1: Protected subspace (REUSING $SUBSPACE_OUT) ==="
else
    echo ""
    echo "=== Step 1: Building protected subspace from GoEmotions ==="
    python scripts/build_protected_subspace.py \
        --model "$MODEL" \
        --emotion_vectors "$EV" \
        --output "$SUBSPACE_OUT" \
        --hidden_cache "$HIDDEN_CACHE" \
        --max_examples 5000 \
        --gamma 0.95 \
        --cv 3 \
        --layers auto
fi

# ---------- Step 2: Bias directions (Stages B+C) ----------
if [ -f "$DIR_OUT" ]; then
    echo ""
    echo "=== Step 2: Mitigation directions (REUSING $DIR_OUT) ==="
    echo "  Delete $DIR_OUT to force recomputation."
else
    echo ""
    echo "=== Step 2: Computing per-dataset + pooled bias directions ==="
    python scripts/compute_mitigation_directions.py \
        --emotion_vectors "$EV" \
        --protected_subspace "$SUBSPACE_OUT" \
        --stereoset_results "$SS_BASE" \
        --genassoc_results "$GA_BASE" \
        --bbq_results "$BBQ_BASE" \
        --output "$DIR_OUT" \
        --layers auto
fi

# ---------- Step 3: Decide target steering layer(s) ----------
if [ -z "$STEER_LAYERS" ]; then
    STEER_LAYERS=$(python - <<PY
import torch
# Read from the CHOSEN SCOPE's actual keys. These are the layers that have
# populated mitigation directions — strictly what probe_with_steering can
# legally use. ( mit["target_layers"] can be larger than scopes[scope]:
# target_layers reflects the INTENDED layer set from the subspace file,
# while scopes[scope] reflects what was ACTUALLY processed -- scopes skip
# a layer if the probe results have no valid projections at it. )
mit = torch.load("$DIR_OUT", weights_only=False)
scope = "$SCOPE"
layers = sorted(mit["scopes"].get(scope, {}).keys())
if not layers:
    for sc, sd in mit["scopes"].items():
        if sd:
            layers = sorted(sd.keys())
            print(f"[warn] scope {scope} has no layers; falling back to {sc}", flush=True)
            break
if not layers:
    raise SystemExit("No populated scopes in directions file.")
# Ideal 2/3-depth pick, then snap to nearest available layer if that index
# doesn't exist in this scope's processed set.
ideal = layers[len(layers) * 2 // 3]
print(ideal)
PY
)
    echo ""
    echo "[auto] Steering layer (2/3 depth, from directions scope='$SCOPE'): $STEER_LAYERS"
fi

COND_FLAG=""
if [ "$CONDITIONAL" = "1" ]; then
    COND_FLAG="--conditional"
    echo "[config] Conditional steering ENABLED"
fi
echo "[config] scope=$SCOPE  method=$METHOD  subspace=$SUBSPACE  alpha=$ALPHA  layers=$STEER_LAYERS  ${COND_FLAG:-unconditional}"

# ---------- Step 4: Run steered probes (Stage D) ----------
echo ""
echo "=== Step 4a: Steered StereoSet ==="
python scripts/probe_with_steering.py \
    --model "$MODEL" \
    --emotion_vectors "$EV" \
    --mitigation_directions "$DIR_OUT" \
    --protected_subspace "$SUBSPACE_OUT" \
    --dataset stereoset \
    --data data/stereoset.json \
    --output "$SS_STEER" \
    --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
    --alpha "$ALPHA" --target_layers "$STEER_LAYERS" $COND_FLAG

echo ""
echo "=== Step 4b: Steered GenAssocBias ==="
python scripts/probe_with_steering.py \
    --model "$MODEL" \
    --emotion_vectors "$EV" \
    --mitigation_directions "$DIR_OUT" \
    --protected_subspace "$SUBSPACE_OUT" \
    --dataset genassoc \
    --data data/GenAssocBias.csv \
    --output "$GA_STEER" \
    --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
    --alpha "$ALPHA" --target_layers "$STEER_LAYERS" $COND_FLAG

echo ""
echo "=== Step 4c: Steered BBQ ==="
python scripts/probe_with_steering.py \
    --model "$MODEL" \
    --emotion_vectors "$EV" \
    --mitigation_directions "$DIR_OUT" \
    --protected_subspace "$SUBSPACE_OUT" \
    --dataset bbq \
    --data data/bbq_all.jsonl \
    --output "$BBQ_STEER" \
    --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
    --alpha "$ALPHA" --target_layers "$STEER_LAYERS" $COND_FLAG \
    --max_new_tokens 12

# ---------- Step 5: Pre/post analysis ----------
echo ""
echo "=== Step 5: Analyzing baseline vs steered ==="
python scripts/analyze_mitigation.py \
    --model_short "$MODEL_SHORT" \
    --output_dir outputs/figures/ \
    --stereoset_baseline "$SS_BASE"   --stereoset_steered "$SS_STEER" \
    --genassoc_baseline  "$GA_BASE"   --genassoc_steered  "$GA_STEER" \
    --bbq_baseline       "$BBQ_BASE"  --bbq_steered       "$BBQ_STEER"

echo ""
echo "=============================================="
echo "Mitigation run complete for: $MODEL"
echo "Config: scope=$SCOPE method=$METHOD subspace=$SUBSPACE alpha=$ALPHA layers=$STEER_LAYERS cond=$CONDITIONAL"
echo "=============================================="
