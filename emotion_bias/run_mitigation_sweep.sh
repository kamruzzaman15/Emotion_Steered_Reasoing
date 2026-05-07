#!/bin/bash
# run_mitigation_sweep.sh -- Alpha sweep with subset/full modes and MMLU capability check.
#
# CACHE KEY FIX
#   Every steered output file now encodes the FULL steering configuration in
#   its filename: layer, alpha, scope, method, subspace, conditional, and
#   max_examples. Change any of those and you get a fresh file — no more
#   silent reuse of stale results across different configs.
#
# SUBSET MODE
#   Set MAX_EX to limit each bias dataset to a fixed number of examples.
#     MAX_EX=2000 ...   ->   2000 examples per dataset (deterministic subsample)
#     (unset/empty)     ->   FULL dataset
#   Baseline rates are computed on the matching subset by pairing example IDs
#   (handled in summarize_sweep.py), so subset comparisons are apples-to-apples.
#
# MMLU CAPABILITY
#   Baseline MMLU is cached per N_MMLU (so different Ns coexist).
#   Steered MMLU uses the SAME alpha/layer/method as the bias probe.
#
# PREREQS (produced by run_mitigation.sh — error if missing):
#   outputs/<slug>_emotion_vectors.pt
#   outputs/<slug>_protected_subspace.pt
#   outputs/<slug>_mitigation_directions.pt
#   outputs/<slug>_stereoset_results.pt   outputs/<slug>_genassoc_results.pt
#   outputs/<slug>_bbq_results.pt
#
# USAGE
#   # Quick-iterate on a subset (2k per dataset, one alpha):
#   MAX_EX=2000 LAYER=25 ALPHAS="360" bash run_mitigation_sweep.sh
#
#   # Same config but on the FULL datasets (no MAX_EX):
#   LAYER=25 ALPHAS="360" bash run_mitigation_sweep.sh
#
#   # Multi-alpha subset sweep:
#   MAX_EX=2000 LAYER=25 ALPHAS="200 360 500" bash run_mitigation_sweep.sh
#
#   # Multi-layer steering:
#   MAX_EX=2000 LAYER="20,24,25" ALPHAS="200" bash run_mitigation_sweep.sh
#
#   # Larger MMLU sample:
#   N_MMLU=2000 MAX_EX=2000 LAYER=25 ALPHAS="360" bash run_mitigation_sweep.sh

set -e
cd "$(dirname "$0")"

MODEL="${1:-google/gemma-2-2b-it}"
SCOPE="${SCOPE:-pooled}"
METHOD="${METHOD:-logreg}"
SUBSPACE="${SUBSPACE:-task}"
LAYER="${LAYER:-}"                 # auto-pick if empty
ALPHAS="${ALPHAS:-360}"
MAX_EX="${MAX_EX:-}"               # empty/unset = full dataset
N_MMLU="${N_MMLU:-500}"
CONDITIONAL="${CONDITIONAL:-0}"

slug=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '.' '_')

EV="outputs/${slug}_emotion_vectors.pt"
SUBSPACE_FILE="outputs/${slug}_protected_subspace.pt"
DIR_FILE="outputs/${slug}_mitigation_directions.pt"
SS_BASE="outputs/${slug}_stereoset_results.pt"
GA_BASE="outputs/${slug}_genassoc_results.pt"
BBQ_BASE="outputs/${slug}_bbq_results.pt"

mkdir -p outputs outputs/figures

# ---------- Prereq check ----------
echo "=== Prereq check ==="
for f in "$EV" "$SUBSPACE_FILE" "$DIR_FILE" "$SS_BASE" "$GA_BASE" "$BBQ_BASE"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        echo "  Run 'bash run_mitigation.sh $MODEL' first to build Stage A+B+C artifacts."
        exit 1
    fi
    echo "  OK: $f"
done

# ---------- Auto-pick steering layer from chosen scope's actual keys ----------
if [ -z "$LAYER" ]; then
    LAYER=$(python - <<PY
import torch
mit = torch.load("$DIR_FILE", weights_only=False)
scope = "$SCOPE"
layers = sorted(mit["scopes"].get(scope, {}).keys())
if not layers:
    for sc, sd in mit["scopes"].items():
        if sd:
            layers = sorted(sd.keys()); break
print(layers[len(layers) * 2 // 3])
PY
)
    echo "[auto] Steering layer (2/3 depth, from directions scope='$SCOPE'): $LAYER"
fi

# ---------- Build tag components for filenames ----------
# layer tag: replace commas with dashes so filenames are safe
LAYER_TAG=$(echo "$LAYER" | tr ',' '-')

if [ "$CONDITIONAL" = "1" ]; then
    COND_FLAG="--conditional"
    COND_TAG="cnd"
else
    COND_FLAG=""
    COND_TAG="unc"
fi

# MAX_EX -> "full" when empty, "NNN" otherwise
if [ -z "$MAX_EX" ] || [ "$MAX_EX" = "0" ]; then
    MAX_EX_TAG="full"
    MAX_EX_ARG=""                  # empty means: don't pass --max_examples (probe uses all)
else
    MAX_EX_TAG="${MAX_EX}"
    MAX_EX_ARG="--max_examples ${MAX_EX}"
fi

echo ""
echo "=== Config ==="
echo "  model       : $MODEL"
echo "  scope       : $SCOPE"
echo "  method      : $METHOD"
echo "  subspace    : $SUBSPACE"
echo "  layer(s)    : $LAYER        (tag: L${LAYER_TAG})"
echo "  alphas      : $ALPHAS"
echo "  conditional : $CONDITIONAL       (tag: ${COND_TAG})"
echo "  max_examples: $MAX_EX_TAG"
echo "  N_MMLU      : $N_MMLU"

# ---------- MMLU baseline (keyed on N_MMLU so different N coexist) ----------
MMLU_BASE="outputs/${slug}_mmlu_baseline_N${N_MMLU}.pt"
if [ ! -f "$MMLU_BASE" ]; then
    echo ""
    echo "============================================================"
    echo "MMLU BASELINE (no steering, N=$N_MMLU)"
    echo "============================================================"
    python scripts/mmlu_eval.py \
        --model "$MODEL" \
        --output "$MMLU_BASE" \
        --n_questions "$N_MMLU"
else
    echo ""
    echo "=== MMLU baseline REUSED from $MMLU_BASE ==="
fi

# ---------- Helper: build full-config filename for a given dataset + alpha ----------
# Produces something like:
#   outputs/<slug>_stereoset_steered_L25_a360_pooled_logreg_task_unc_N2000.pt
build_name() {
    local kind="$1"            # "stereoset" | "genassoc" | "bbq" | "mmlu"
    local alpha_int="$2"
    if [ "$kind" = "mmlu" ]; then
        echo "outputs/${slug}_mmlu_L${LAYER_TAG}_a${alpha_int}_${SCOPE}_${METHOD}_${SUBSPACE}_${COND_TAG}_N${N_MMLU}.pt"
    else
        echo "outputs/${slug}_${kind}_steered_L${LAYER_TAG}_a${alpha_int}_${SCOPE}_${METHOD}_${SUBSPACE}_${COND_TAG}_N${MAX_EX_TAG}.pt"
    fi
}

# Compact alpha tag: 360.0 -> 360, 3.5 -> 3_5
alpha_tag() {
    python -c "
a = float('$1')
print(int(a) if a.is_integer() else str(a).replace('.', '_'))
"
}

# ---------- Sweep over alphas ----------
for ALPHA in $ALPHAS; do
    ATAG=$(alpha_tag "$ALPHA")
    SS_STEER=$(build_name stereoset "$ATAG")
    GA_STEER=$(build_name genassoc  "$ATAG")
    BBQ_STEER=$(build_name bbq       "$ATAG")
    MMLU_STEER=$(build_name mmlu     "$ATAG")

    echo ""
    echo "############################################################"
    echo "### alpha = $ALPHA   layer = $LAYER   max_examples = $MAX_EX_TAG"
    echo "###   stereo : $SS_STEER"
    echo "###   genassoc: $GA_STEER"
    echo "###   bbq    : $BBQ_STEER"
    echo "###   mmlu   : $MMLU_STEER"
    echo "############################################################"

    if [ ! -f "$SS_STEER" ]; then
        echo ""
        echo "--- StereoSet (steered) ---"
        python scripts/probe_with_steering.py \
            --model "$MODEL" --emotion_vectors "$EV" \
            --mitigation_directions "$DIR_FILE" --protected_subspace "$SUBSPACE_FILE" \
            --dataset stereoset --data data/stereoset.json --output "$SS_STEER" \
            --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
            --alpha "$ALPHA" --target_layers "$LAYER" $COND_FLAG $MAX_EX_ARG
    else
        echo "--- StereoSet REUSED: $SS_STEER ---"
    fi

    if [ ! -f "$GA_STEER" ]; then
        echo ""
        echo "--- GenAssocBias (steered) ---"
        python scripts/probe_with_steering.py \
            --model "$MODEL" --emotion_vectors "$EV" \
            --mitigation_directions "$DIR_FILE" --protected_subspace "$SUBSPACE_FILE" \
            --dataset genassoc --data data/GenAssocBias.csv --output "$GA_STEER" \
            --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
            --alpha "$ALPHA" --target_layers "$LAYER" $COND_FLAG $MAX_EX_ARG
    else
        echo "--- GenAssoc REUSED: $GA_STEER ---"
    fi

    if [ ! -f "$BBQ_STEER" ]; then
        echo ""
        echo "--- BBQ (steered) ---"
        python scripts/probe_with_steering.py \
            --model "$MODEL" --emotion_vectors "$EV" \
            --mitigation_directions "$DIR_FILE" --protected_subspace "$SUBSPACE_FILE" \
            --dataset bbq --data data/bbq_all.jsonl --output "$BBQ_STEER" \
            --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
            --alpha "$ALPHA" --target_layers "$LAYER" $COND_FLAG $MAX_EX_ARG \
            --max_new_tokens 12
    else
        echo "--- BBQ REUSED: $BBQ_STEER ---"
    fi

    if [ ! -f "$MMLU_STEER" ]; then
        echo ""
        echo "--- MMLU (steered) ---"
        python scripts/mmlu_eval.py \
            --model "$MODEL" --output "$MMLU_STEER" --n_questions "$N_MMLU" \
            --emotion_vectors "$EV" --mitigation_directions "$DIR_FILE" \
            --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
            --alpha "$ALPHA" --target_layers "$LAYER" $COND_FLAG
    else
        echo "--- MMLU REUSED: $MMLU_STEER ---"
    fi
done

# ---------- Rich combined summary ----------
MODEL_SHORT=$(python -c "from scripts.analyze_results import shorten_slug; print(shorten_slug('$slug'))")
CSV_OUT="outputs/figures/${MODEL_SHORT}_sweep_summary_L${LAYER_TAG}_${COND_TAG}_N${MAX_EX_TAG}.csv"

echo ""
echo "############################################################"
echo "### COMBINED BEFORE / AFTER SUMMARY"
echo "############################################################"
python scripts/summarize_sweep.py \
    --model "$MODEL" --slug "$slug" \
    --scope "$SCOPE" --method "$METHOD" --subspace "$SUBSPACE" \
    --layer "$LAYER" --alphas "$ALPHAS" \
    --conditional "$CONDITIONAL" \
    --max_examples_tag "$MAX_EX_TAG" \
    --n_mmlu "$N_MMLU" \
    --ss_base "$SS_BASE" --ga_base "$GA_BASE" --bbq_base "$BBQ_BASE" \
    --mmlu_base "$MMLU_BASE" \
    --emotion_vectors "$EV" --mitigation_directions "$DIR_FILE" \
    --output "$CSV_OUT"

echo ""
echo "============================================================"
echo "Done."
echo "  Summary CSV: $CSV_OUT"
echo "============================================================"
