import os
import json
import sys
import base64
import io
import csv
import requests
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("data/mmmb", exist_ok=True)
os.makedirs("data/mmmb/images", exist_ok=True)

MMMB_LANGS = ["en", "zh", "ar"]

# Source: AIDC-AI/Parrot-dataset on HuggingFace
# Files confirmed at: .../resolve/main/mmmb/mmmb_{file}.tsv
TSV_FILES = {
    "en": "mmmb_en",
    "zh": "mmmb_cn",
    "ar": "mmmb_ar",
}
BASE_URL = "https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb"


def decode_image(b64_str, out_path):
    """Decode raw base64 image string and save to out_path. Returns path or None."""
    try:
        img_bytes = base64.b64decode(b64_str)
        PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB").save(out_path)
        return out_path
    except Exception as e:
        print(f"  Warning: could not decode image → {out_path}: {e}")
        return None


for lang in MMMB_LANGS:
    out_path = f"data/mmmb/mmmb_{lang}.json"
    tsv_name = TSV_FILES[lang]
    url = f"{BASE_URL}/{tsv_name}.tsv"

    print(f"\n[{lang}] Downloading {url} ...")
    try:
        r = requests.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        print(f"  Download failed: {e}")
        json.dump([], open(out_path, "w"))
        continue

    reader = csv.DictReader(io.StringIO(r.content.decode("utf-8")), delimiter="\t")
    examples = []
    for idx, row in enumerate(reader):
        if idx >= 200:
            break

        # Question — prepend hint if present (provides visual context)
        question = row.get("question", "").strip()
        hint = row.get("hint", "").strip()
        if hint:
            question = f"{hint}\n{question}"

        # Options — A/B/C/D/E (keep only non-empty ones)
        options = {k: row.get(k, "").strip() for k in ["A", "B", "C", "D", "E"]}
        options = {k: v for k, v in options.items() if v}

        # Answer
        answer = row.get("answer", "A").strip()
        if len(answer) > 1:
            answer = answer[0]

        # Image — raw base64 in 'image' column
        img_path = ""
        raw_b64 = row.get("image", "").strip()
        if raw_b64:
            img_out = f"data/mmmb/images/{lang}_{idx:04d}.jpg"
            img_path = decode_image(raw_b64, img_out) or ""

        examples.append({
            "question": question,
            "options":  options,
            "answer":   answer,
            "image":    img_path,
        })

    json.dump(examples, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    non_empty_q   = sum(1 for e in examples if e["question"].strip())
    non_empty_img = sum(1 for e in examples if e["image"])
    print(f"  [{lang}] {len(examples)} examples  "
          f"non-empty question={non_empty_q}  image={non_empty_img}  → {out_path}")

print("\nMMMB download summary:")
for lang in MMMB_LANGS:
    path = f"data/mmmb/mmmb_{lang}.json"
    if os.path.exists(path):
        data = json.load(open(path))
        print(f"  {lang}: {len(data)} examples")
    else:
        print(f"  {lang}: missing")
