#!/bin/bash
# run_all.sh — Full multi-model pipeline for bias evaluation
# Usage: bash run_all.sh [--quick]

set -e

QUICK=false
if [[ "$1" == "--quick" ]]; then
    QUICK=true
    echo "=== QUICK MODE ==="
fi

cd "$(dirname "$0")"

# ============================================================
# Models to evaluate (one at a time, cleaned up between runs)
# ============================================================
MODELS=(
    "google/gemma-2-2b-it"
    "google/gemma-3-1b-it"
    "meta-llama/Llama-3.2-3B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "allenai/OLMo-2-1124-7B"
)

# ============================================================
# Step 0: Environment Setup and Dataset Consolidation
# ============================================================
echo "=== Step 0: Install dependencies and verify datasets ==="
pip install -r requirements.txt --quiet

# Fix: Consolidate BBQ categories without "input file is output file" error
echo "  Consolidating BBQ categories..."
rm -f data/bbq_all.jsonl  # Remove old version so wildcard doesn't catch it
cat data/*.jsonl > data/bbq_all.jsonl

# Verify all required local datasets exist
if [ ! -f "data/stereoset.json" ] || [ ! -f "data/bbq_all.jsonl" ] || [ ! -f "data/GenAssocBias.csv" ] || [ ! -f "data/bold.json" ]; then
    echo "  Error: Required datasets (stereoset, bbq_all, GenAssocBias, or bold) missing in 'data/'."
    exit 1
else
    echo "  Datasets verified. BBQ master file lines: $(wc -l < data/bbq_all.jsonl)"
fi

# ============================================================
# Pipeline Loop
# ============================================================
for MODEL in "${MODELS[@]}"; do
    slug=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '.' '_')
    echo ""
    echo "**********************************************"
    echo "RUNNING PIPELINE FOR: $MODEL"
    echo "**********************************************"

    # Define sample sizes and story generation limits
    if [ "$QUICK" = true ]; then
        SS_SAMPLES=100
        GEN_SAMPLES=100
        BBQ_SAMPLES=100
        BOLD_SAMPLES=20
        NUM_STORIES=1      # 2 stories per emotion for quick testing
    else
        SS_SAMPLES=0       # 0 = Use entire dataset
        GEN_SAMPLES=0      # 0 = Use entire dataset
        BBQ_SAMPLES=0      # 0 = Use entire dataset
        BOLD_SAMPLES=0     # 0 = Use entire dataset
        NUM_STORIES=50     # 20 stories per emotion for research run
    fi

    # Step 1: Extract Emotion Vectors
    echo "--- Step 1: Extracting Emotion Vectors ($NUM_STORIES stories/emotion) ---"
    python scripts/extract_emotion_vectors.py \
        --model "$MODEL" \
        --output "outputs/${slug}_emotion_vectors.pt" \
        --stories_output "outputs/${slug}_generated_stories.json" \
        --num_stories $NUM_STORIES

    # Step 2: Probing StereoSet
    echo "--- Step 2: Probing StereoSet ---"
    python scripts/probe_stereoset.py \
        --model "$MODEL" \
        --emotion_vectors "outputs/${slug}_emotion_vectors.pt" \
        --data "data/stereoset.json" \
        --output "outputs/${slug}_stereoset_results.pt" \
        --max_examples $SS_SAMPLES

    # Step 3: Probing GenAssocBias (Matched to your local CSV)
    echo "--- Step 3: Probing GenAssocBias ---"
    python scripts/probe_genassocbias.py \
        --model "$MODEL" \
        --emotion_vectors "outputs/${slug}_emotion_vectors.pt" \
        --data "data/GenAssocBias.csv" \
        --output "outputs/${slug}_genassoc_results.pt" \
        --max_examples $GEN_SAMPLES \
        --layers "quarter"

    # Step 4: Probing BBQ
    echo "--- Step 4: Probing BBQ ---"
    python scripts/probe_bbq.py \
        --model "$MODEL" \
        --emotion_vectors "outputs/${slug}_emotion_vectors.pt" \
        --data "data/bbq_all.jsonl" \
        --output "outputs/${slug}_bbq_results.pt" \
        --max_examples $BBQ_SAMPLES \
        --max_new_tokens 12 \
        --layers "quarter"

    # Step 5: Probing BOLD (Uses corrected --max_examples argument)
    echo "--- Step 5: Probing BOLD ---"
    python scripts/probe_bold.py \
        --model "$MODEL" \
        --emotion_vectors "outputs/${slug}_emotion_vectors.pt" \
        --data "data/bold.json" \
        --output "outputs/${slug}_bold_results.pt" \
        --max_examples $BOLD_SAMPLES

    # Step 6: Analysis and Figures
    echo "--- Step 6: Analyzing results and generating figures ---"
    MODEL_SHORT=$(python -c "from scripts.analyze_results import shorten_slug; print(shorten_slug('$slug'))")
    
    python scripts/analyze_results.py \
        --model_slug "$slug" \
        --model_short "$MODEL_SHORT" \
        --stereoset "outputs/${slug}_stereoset_results.pt" \
        --genassocbias "outputs/${slug}_genassoc_results.pt" \
        --bbq "outputs/${slug}_bbq_results.pt" \
        --bold "outputs/${slug}_bold_results.pt" \
        --output "outputs/figures/"

    # Step 7: Cleanup model cache to save disk space
    echo "--- Step 7: Cleanup model weights for $MODEL ---"
    CACHE_NAME="models--$(echo "$MODEL" | tr '/' '--')"
    CACHE_DIR="$HOME/.cache/huggingface/hub/${CACHE_NAME}"
    
    if [ -d "$CACHE_DIR" ]; then
        rm -rf "$CACHE_DIR"
        echo "  Successfully removed model weights."
    else
        MODEL_BASENAME=$(basename "$MODEL")
        for match in "$HOME/.cache/huggingface/hub/"models--*${MODEL_BASENAME}*; do
            if [ -d "$match" ]; then
                rm -rf "$match"
                echo "  Found and removed shard: $match"
            fi
        done
    fi

    # Release GPU memory for the next model
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()"
    
    echo "=== Completed: $MODEL ==="
done

echo "=============================================="
echo "All models complete!"
echo "=============================================="