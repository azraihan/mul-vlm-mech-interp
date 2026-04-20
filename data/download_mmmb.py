import os
import json
import sys
import base64
import io
import requests
import numpy as np
from tqdm import tqdm
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("data/mmmb", exist_ok=True)

MMMB_LANGS = ["en", "zh", "ar"]

# STEP 1 & 2: Try HuggingFace
ds = None
hf_success = False

for hf_id in ["Awiny/MMBench_MMMB", "HuggingFaceM4/MMBench_MMMB"]:
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_id)
        hf_success = True
        print(f"Loaded MMMB from HuggingFace ({hf_id})")
        break
    except Exception as e:
        print(f"HuggingFace MMMB ({hf_id}) failed: {e}")

if not hf_success:
    print("Warning: MMMB not available from HuggingFace. Attempting MMBench TSV fallback.")

# Helper: save image from dataset field
def save_image_field(img_field, out_path):
    """Save image from various formats to file. Returns out_path or None."""
    try:
        if isinstance(img_field, PIL.Image.Image):
            img_field.save(out_path)
            return out_path
        elif isinstance(img_field, np.ndarray):
            PIL.Image.fromarray(img_field).save(out_path)
            return out_path
        elif isinstance(img_field, bytes):
            PIL.Image.open(io.BytesIO(img_field)).save(out_path)
            return out_path
        elif isinstance(img_field, str) and img_field.startswith("data:image"):
            # base64 data URI
            b64 = img_field.split(",", 1)[1]
            PIL.Image.open(io.BytesIO(base64.b64decode(b64))).save(out_path)
            return out_path
    except Exception as e:
        print(f"  Warning: could not save image to {out_path}: {e}")
    return None


def parse_hf_example(ex, lang, idx):
    """Parse a HuggingFace dataset example into our format."""
    question = ex.get("question", "")
    # Try to get options
    options = {}
    for letter in ["A", "B", "C", "D"]:
        opt_key = f"option_{letter}" if f"option_{letter}" in ex else letter.lower()
        if opt_key in ex:
            options[letter] = str(ex[opt_key])
        elif letter in ex:
            options[letter] = str(ex[letter])
        else:
            options[letter] = ""

    # Try common answer key names
    answer = str(ex.get("answer", ex.get("gt_answer", ex.get("target", "A"))))
    if len(answer) > 1:
        answer = answer[0]  # take first char if "A. ..." style

    # Image
    img_field = ex.get("image", ex.get("img", None))
    img_path = None
    if img_field is not None:
        img_out = f"data/mmmb/images/{lang}_{idx:04d}.jpg"
        os.makedirs("data/mmmb/images", exist_ok=True)
        img_path = save_image_field(img_field, img_out)

    return {
        "question": question,
        "options": options,
        "answer": answer,
        "image": img_path if img_path else "",
    }


# STEP 3: For each language, save 200 examples
if hf_success and ds is not None:
    # Try to figure out splits
    available_splits = list(ds.keys())
    print(f"  Available splits: {available_splits}")

    for lang in MMMB_LANGS:
        out_path = f"data/mmmb/mmmb_{lang}.json"
        examples = []

        # Common split naming patterns
        candidate_splits = [
            lang, f"test_{lang}", f"dev_{lang}", f"{lang}_test", f"{lang}_dev",
            "test", "dev", "train", "validation",
            "en" if lang == "en" else None,
        ]
        split_data = None
        for split_name in candidate_splits:
            if split_name and split_name in available_splits:
                split_data = ds[split_name]
                print(f"  [{lang}] using split '{split_name}'")
                break

        if split_data is None and available_splits:
            split_data = ds[available_splits[0]]
            print(f"  [{lang}] falling back to split '{available_splits[0]}'")

        if split_data is not None:
            for idx, ex in enumerate(split_data):
                if idx >= 200:
                    break
                parsed = parse_hf_example(dict(ex), lang, idx)
                examples.append(parsed)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"MMMB [{lang}]: {len(examples)} examples → {out_path}")

else:
    # Fallback: try TSV from opencompass (en and zh only; ar likely unavailable)
    print("Trying MMBench TSV fallback (en + zh)...")
    import csv

    tsv_map = {
        "en": "https://opencompass.org.cn/MMBench/MMBench_DEV_EN.tsv",
        "zh": "https://opencompass.org.cn/MMBench/MMBench_DEV_CN.tsv",
    }

    for lang in MMMB_LANGS:
        out_path = f"data/mmmb/mmmb_{lang}.json"
        examples = []

        if lang == "ar":
            print(f"  Warning: no Arabic MMMB split available. Skipping ar.")
            with open(out_path, "w") as f:
                json.dump([], f)
            continue

        url = tsv_map.get(lang)
        if url is None:
            continue

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
            for idx, row in enumerate(reader):
                if idx >= 200:
                    break
                options = {
                    "A": row.get("A", ""),
                    "B": row.get("B", ""),
                    "C": row.get("C", ""),
                    "D": row.get("D", ""),
                }
                # Image may be base64 in 'image' column
                img_path = ""
                if "image" in row and row["image"]:
                    img_out = f"data/mmmb/images/{lang}_{idx:04d}.jpg"
                    os.makedirs("data/mmmb/images", exist_ok=True)
                    img_path = save_image_field(row["image"], img_out) or ""
                examples.append({
                    "question": row.get("question", ""),
                    "options": options,
                    "answer": row.get("answer", "A"),
                    "image": img_path,
                })
        except Exception as e:
            print(f"  TSV download for {lang} failed: {e}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"MMMB [{lang}]: {len(examples)} examples → {out_path}")

# STEP 4: Print summary
print("\nMMMB download summary:")
for lang in MMMB_LANGS:
    path = f"data/mmmb/mmmb_{lang}.json"
    if os.path.exists(path):
        data = json.load(open(path))
        print(f"  {lang}: {len(data)} examples")
    else:
        print(f"  {lang}: missing")
