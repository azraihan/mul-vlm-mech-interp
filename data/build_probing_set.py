import os
import json
import sys
import zipfile
import requests
from collections import defaultdict
from tqdm import tqdm
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import load_model, get_token_id
from models.constants import CATEGORIES, TRANSLATIONS, QUESTIONS

os.makedirs("data/probing_set/images", exist_ok=True)


def download_file(url, dest_path, desc="Downloading"):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


# STEP 2: Download COCO val2017 if not already present
if not os.path.exists("data/coco_val2017/images/000000000139.jpg"):
    os.makedirs("data/coco_val2017/images", exist_ok=True)
    os.makedirs("data/coco_val2017/annotations", exist_ok=True)

    try:
        img_zip = "data/coco_val2017/val2017.zip"
        ann_zip = "data/coco_val2017/annotations.zip"

        if not os.path.exists(img_zip):
            download_file(
                "http://images.cocodataset.org/zips/val2017.zip",
                img_zip,
                "COCO val2017 images"
            )
        print("Extracting COCO images...")
        with zipfile.ZipFile(img_zip, "r") as zf:
            for member in tqdm(zf.namelist(), desc="Extracting images"):
                if member.startswith("val2017/") and member.endswith(".jpg"):
                    fname = os.path.basename(member)
                    out_path = f"data/coco_val2017/images/{fname}"
                    if not os.path.exists(out_path):
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())

        if not os.path.exists(ann_zip):
            download_file(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                ann_zip,
                "COCO annotations"
            )
        print("Extracting COCO annotations...")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            for member in zf.namelist():
                if "instances_val2017.json" in member:
                    with zf.open(member) as src:
                        with open("data/coco_val2017/annotations/instances_val2017.json", "wb") as dst:
                            dst.write(src.read())
                    break

    except Exception as e:
        print(f"COCO download failed: {e}")
        print("Trying HuggingFace fallback...")
        from datasets import load_dataset
        ds = load_dataset("detection-datasets/coco", split="validation")
        os.makedirs("data/coco_val2017/images", exist_ok=True)
        for i, ex in enumerate(tqdm(ds, desc="Saving COCO images")):
            img = PIL.Image.fromarray(ex["image"]) if not isinstance(ex["image"], PIL.Image.Image) else ex["image"]
            fname = f"{ex['image_id']:012d}.jpg"
            img.save(f"data/coco_val2017/images/{fname}")
        # Build a synthetic annotations file from HF dataset
        categories = []
        annotations = []
        images = []
        ann_id = 0
        # Extract category names from dataset features
        try:
            cats_raw = ds.features["objects"].feature["category"].names
        except Exception:
            cats_raw = []
        for ci, cname in enumerate(cats_raw):
            categories.append({"id": ci, "name": cname})
        # Single pass: collect images + annotations
        for ex in ds:
            images.append({"id": ex["image_id"], "file_name": f"{ex['image_id']:012d}.jpg"})
            objs = ex.get("objects", {})
            if isinstance(objs, dict):
                for bbox, cat_id in zip(objs.get("bbox", []), objs.get("category", [])):
                    x, y, w, h = bbox
                    annotations.append({
                        "id": ann_id,
                        "image_id": ex["image_id"],
                        "category_id": cat_id,
                        "bbox": [x, y, w, h],
                    })
                    ann_id += 1
        with open("data/coco_val2017/annotations/instances_val2017.json", "w") as f:
            json.dump({"categories": categories, "annotations": annotations, "images": images}, f)

# STEP 3: Load COCO annotations
with open("data/coco_val2017/annotations/instances_val2017.json") as f:
    data = json.load(f)

coco_cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
image_id_to_annotations = defaultdict(list)
for ann in data["annotations"]:
    image_id_to_annotations[ann["image_id"]].append(ann)
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# STEP 4: Load model and processor for tokenizer verification
model, processor = load_model()

# STEP 5: Build the probing set
examples = []
for category in CATEGORIES:
    # 5a: Verify all 5 language translations tokenize to exactly 1 token
    token_ids = {}
    valid = True
    for lang in ["en", "fr", "ar", "zh", "bn"]:
        word = TRANSLATIONS[category][lang]
        tid = get_token_id(processor, word)
        if tid is None:
            print(f"SKIP category={category} lang={lang}: '{word}' is multi-token")
            valid = False
            break
        token_ids[lang] = tid
    if not valid:
        continue

    # 5b: Find top-4 COCO images by bounding box area
    coco_cat_id = coco_cat_name_to_id.get(category)
    if coco_cat_id is None:
        print(f"SKIP category={category}: not in COCO")
        continue
    anns = [a for a in data["annotations"] if a["category_id"] == coco_cat_id]
    anns_sorted = sorted(anns, key=lambda a: a["bbox"][2] * a["bbox"][3], reverse=True)
    seen_ids = []
    for ann in anns_sorted:
        if ann["image_id"] not in seen_ids:
            seen_ids.append(ann["image_id"])
        if len(seen_ids) == 4:
            break

    # 5c: Copy + resize each image; build example dict
    for i, image_id in enumerate(seen_ids):
        src_filename = image_id_to_filename[image_id]
        src_path = f"data/coco_val2017/images/{src_filename}"
        dst_path = f"data/probing_set/images/{category}_{i:03d}.jpg"
        img = PIL.Image.open(src_path).convert("RGB").resize((336, 336))
        img.save(dst_path)
        examples.append({
            "image_path": dst_path,
            "category": category,
            "answers":   {lang: TRANSLATIONS[category][lang]
                          for lang in ["en", "fr", "ar", "zh", "bn"]},
            "token_ids": token_ids,
            "questions": QUESTIONS,
        })

# STEP 6: Save and report
with open("data/probing_set/probing_set.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)
print(f"Probing set: {len(examples)} examples across {len(set(e['category'] for e in examples))} categories")
assert len(examples) >= 40, "Too few examples — check tokenizer filtering"
