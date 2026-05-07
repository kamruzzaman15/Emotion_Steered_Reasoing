"""
download_datasets.py - Download and prepare StereoSet, BBQ, and BOLD datasets.
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm


def download_stereoset(output_dir: str):
    """Download and format StereoSet intrasentence examples."""
    print("Downloading StereoSet...")
    ds = load_dataset("McGill-NLP/stereoset", "intersentence")

    processed = []
    for item in tqdm(ds["validation"], desc="Processing StereoSet"):
        context = item["context"]
        bias_type = item["bias_type"]
        target = item["target"]

        sentences = item["sentences"]
        stereo_sent, anti_stereo_sent, unrelated_sent = None, None, None

        for sent, gold_label in zip(sentences["sentence"], sentences["gold_label"]):
            if gold_label == 0:  # stereotype
                stereo_sent = sent
            elif gold_label == 1:  # anti-stereotype
                anti_stereo_sent = sent
            elif gold_label == 2:  # unrelated
                unrelated_sent = sent

        if stereo_sent and anti_stereo_sent and unrelated_sent:
            processed.append({
                "id": f"stereoset_{len(processed)}",
                "context": context,
                "stereotype": stereo_sent,
                "anti_stereotype": anti_stereo_sent,
                "unrelated": unrelated_sent,
                "bias_type": bias_type,
                "target": target,
            })

    out_path = os.path.join(output_dir, "stereoset.json")
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"StereoSet: {len(processed)} examples saved to {out_path}")
    return processed


def download_bbq(output_dir: str):
    """Download and format BBQ dataset."""
    print("Downloading BBQ...")

    # BBQ repos organize by bias category, not train/test splits.
    # We load each category and merge.
    from datasets import get_dataset_config_names, concatenate_datasets

    all_ds = []
    repo = None

    # Try Elfsong/BBQ first (parquet, no scripts)
    for candidate_repo in ["Elfsong/BBQ", "walledai/BBQ"]:
        try:
            configs = get_dataset_config_names(candidate_repo)
            print(f"  Using {candidate_repo}, configs: {configs}")
            repo = candidate_repo
            break
        except Exception as e:
            print(f"  {candidate_repo} failed: {e}")
            continue

    if repo:
        # Load each category split
        for config in configs:
            try:
                ds_part = load_dataset(repo, config)
                # Get whichever split is available
                for split_name in ds_part.keys():
                    all_ds.append((config, ds_part[split_name]))
                    print(f"    Loaded {config}/{split_name}: {len(ds_part[split_name])} examples")
            except Exception as e:
                print(f"    Skipping {config}: {e}")
                continue
    else:
        raise RuntimeError("Could not load BBQ from any source. Install latest `datasets` or download manually from https://github.com/nyu-mll/BBQ")

    # Process all loaded data
    processed = []
    for config_name, ds in all_ds:
        sample = ds[0]
        available_fields = set(sample.keys())

        # Auto-detect field names
        if "ans0" in available_fields:
            ans_keys = ("ans0", "ans1", "ans2")
        elif "answer_0" in available_fields:
            ans_keys = ("answer_0", "answer_1", "answer_2")
        else:
            ans_keys = sorted([k for k in available_fields if "ans" in k.lower()])[:3]
            if len(ans_keys) < 3:
                print(f"    WARNING: Cannot find answer fields in {config_name}, skipping. Fields: {available_fields}")
                continue

        context_key = "context" if "context" in available_fields else list(available_fields)[0]
        question_key = "question" if "question" in available_fields else "question"
        label_key = "label" if "label" in available_fields else "correct_label"
        condition_key = "context_condition" if "context_condition" in available_fields else None

        # Derive category from config name (e.g., "race_ethnicity" or "Age_ambig")
        category = config_name.replace("_ambig", "").replace("_disambig", "").lower()
        is_ambig = "ambig" in config_name.lower() and "disambig" not in config_name.lower()

        for item in ds:
            entry = {
                "id": f"bbq_{len(processed)}",
                "context": str(item.get(context_key, "")),
                "question": str(item.get(question_key, "")),
                "answer_0": str(item.get(ans_keys[0], "")),
                "answer_1": str(item.get(ans_keys[1], "")),
                "answer_2": str(item.get(ans_keys[2], "")),
                "correct_label": item.get(label_key, -1),
                "context_condition": str(item.get(condition_key, "ambig" if is_ambig else "disambig")) if condition_key else ("ambig" if is_ambig else "disambig"),
                "category": str(item.get("category", category)),
            }
            processed.append(entry)

    out_path = os.path.join(output_dir, "bbq.json")
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"BBQ: {len(processed)} examples saved to {out_path}")
    return processed


def download_genassocbias(output_dir: str):
    """Download and format GenAssocBias dataset."""
    print("Downloading GenAssocBias...")
    ds = load_dataset("mozaman36/GenAssocBias", split="train")

    processed = []
    for item in tqdm(ds, desc="Processing GenAssocBias"):
        context = item["context"]
        stereo_word = item["stereotype"]
        anti_word = item["anti_stereotype"]
        unrelated_word = item["unrelated"]

        # Build full sentences by replacing BLANK
        stereo_sent = context.replace("BLANK", stereo_word)
        anti_sent = context.replace("BLANK", anti_word)
        unrelated_sent = context.replace("BLANK", unrelated_word)

        processed.append({
            "id": f"genassocbias_{len(processed)}",
            "context": context,
            "stereotype": stereo_sent,
            "anti_stereotype": anti_sent,
            "unrelated": unrelated_sent,
            "bias_type": item["bias_type"],
            "target_gender": item.get("target_gender", "not_specified"),
            "item_category": item.get("item_category", ""),
            "type_category": item.get("type_category", ""),
            "target": item.get("bias_type", ""),  # for StereoSet API compatibility
        })

    out_path = os.path.join(output_dir, "genassocbias.json")
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"GenAssocBias: {len(processed)} examples saved to {out_path}")
    return processed


def download_bold(output_dir: str):
    """Download and format BOLD dataset from GitHub (HF repo is unreliable)."""
    print("Downloading BOLD...")
    import urllib.request

    # BOLD is stored as JSON files on GitHub: amazon-science/bold
    github_base = "https://raw.githubusercontent.com/amazon-science/bold/main/prompts"
    bold_files = {
        "gender": "gender_prompt.json",
        "race": "race_prompt.json",
        "religion": "religious_ideology_prompt.json",
        "political": "political_ideology_prompt.json",
        "profession": "profession_prompt.json",
    }

    processed = []
    for domain, filename in bold_files.items():
        url = f"{github_base}/{filename}"
        print(f"  Downloading {domain} from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode("utf-8"))

            # BOLD JSON structure: { "category_name": { "entity_name": ["prompt1", "prompt2", ...] } }
            for category, entities in data.items():
                for entity_name, prompts in entities.items():
                    if isinstance(prompts, list) and prompts:
                        processed.append({
                            "id": f"bold_{len(processed)}",
                            "domain": domain,
                            "name": entity_name,
                            "category": category,
                            "prompts": prompts,
                        })
            print(f"    {domain}: loaded {sum(len(e) for e in data.values())} entities")
        except Exception as e:
            print(f"    WARNING: Failed to download {domain}: {e}")

    out_path = os.path.join(output_dir, "bold.json")
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"BOLD: {len(processed)} entries saved to {out_path}")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    download_stereoset(args.output_dir)
    download_genassocbias(args.output_dir)
    download_bbq(args.output_dir)
    download_bold(args.output_dir)
    print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    main()
