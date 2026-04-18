# Does Bias Feel Right? Emotion Circuit Activation During Stereotyped Reasoning in LLMs

## Quick Start

```bash
# 1. Clone / copy this directory to your server
# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python scripts/download_datasets.py

# 4. Extract emotion vectors from Gemma-2-2B
python scripts/extract_emotion_vectors.py --model google/gemma-2-2b-it --output outputs/emotion_vectors.pt

# 5. Run bias probing on all three datasets
python scripts/probe_stereoset.py --model google/gemma-2-2b-it --emotion_vectors outputs/emotion_vectors.pt --output outputs/stereoset_results.pt
python scripts/probe_bbq.py --model google/gemma-2-2b-it --emotion_vectors outputs/emotion_vectors.pt --output outputs/bbq_results.pt
python scripts/probe_bold.py --model google/gemma-2-2b-it --emotion_vectors outputs/emotion_vectors.pt --output outputs/bold_results.pt

# 6. Run analysis and generate figures
python scripts/analyze_results.py --stereoset outputs/stereoset_results.pt --bbq outputs/bbq_results.pt --bold outputs/bold_results.pt --output outputs/figures/

# Or run everything at once:
bash run_all.sh
```

## Project Structure
```
does_bias_feel_right/
├── README.md
├── requirements.txt
├── run_all.sh
├── configs/
│   └── emotion_concepts.json      # 30 emotion concepts + story prompts
├── scripts/
│   ├── download_datasets.py       # Download StereoSet, BBQ, BOLD
│   ├── extract_emotion_vectors.py # Extract emotion vectors from model
│   ├── model_utils.py             # Shared model loading + activation extraction
│   ├── probe_stereoset.py         # Probe emotion activations on StereoSet
│   ├── probe_bbq.py               # Probe emotion activations on BBQ
│   ├── probe_bold.py              # Probe emotion activations on BOLD
│   └── analyze_results.py         # Statistical analysis + figure generation
├── data/                          # Downloaded datasets go here
└── outputs/                       # Results, vectors, figures
```

## Hardware Requirements
- GPU with >= 16GB VRAM (A100, V100, RTX 4090, etc.)
- ~20GB disk for model + datasets
- Gemma-2-2B fits in fp16 on a single 16GB GPU

## Expected Runtime
- Emotion vector extraction: ~20 minutes
- StereoSet probing: ~30 minutes
- BBQ probing: ~45 minutes
- BOLD probing: ~60 minutes
- Analysis: ~5 minutes
