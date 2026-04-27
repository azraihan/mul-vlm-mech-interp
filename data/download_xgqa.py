import os
import io
import json
import sys
import requests
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

XGQA_LANGS = ["en", "zh", "bn"]


def download_gqa_image(image_id):
    """Return a GQA image by ID, using data/gqa_images/ cache."""
    os.makedirs("data/gqa_images", exist_ok=True)
    cache_path = f"data/gqa_images/{image_id}.jpg"
    if os.path.exists(cache_path):
        return PIL.Image.open(cache_path).convert("RGB")
    # Stanford hosting is offline — images must be pre-cached by download_xgqa.py
    print(f"  Warning: image {image_id} not in cache and Stanford VG_100K is offline.")
    return None


if __name__ == "__main__":
    os.makedirs("data/xgqa", exist_ok=True)
    os.makedirs("data/gqa_images", exist_ok=True)

    from datasets import load_dataset

    # Primary source: floschne/xgqa — has text + embedded images, image_id in n-prefix format
    print("Loading floschne/xgqa (streaming)...")
    try:
        ds = load_dataset("floschne/xgqa", streaming=True)
        available = list(ds.keys())
        print(f"  Available splits: {available}")
    except Exception as e:
        print(f"floschne/xgqa failed: {e}")
        ds = None

    # Build en answer lookup so non-English splits can include answer_en for fair eval
    en_answers = {}  # question_id -> english answer
    if ds is not None and "en" in ds:
        for ex in ds["en"]:
            qid = str(ex.get("question_id", ""))
            if qid:
                en_answers[qid] = str(ex.get("answer", ex.get("full_answer", ""))).lower().strip()

    for lang in XGQA_LANGS:
        out_path = f"data/xgqa/xgqa_{lang}.json"
        examples = []
        images_saved = 0

        if ds is not None and lang in ds:
            for ex in ds[lang]:
                if len(examples) >= 200:
                    break

                image_id  = str(ex.get("image_id", ""))
                question  = ex.get("question", "")
                answer    = ex.get("answer", ex.get("full_answer", ""))
                qid       = str(ex.get("question_id", ""))

                # Save image to cache if not already there
                cache_path = f"data/gqa_images/{image_id}.jpg"
                if not os.path.exists(cache_path):
                    img_field = ex.get("image")
                    if img_field is not None:
                        try:
                            raw = img_field if isinstance(img_field, bytes) else img_field.get("bytes", b"")
                            PIL.Image.open(io.BytesIO(raw)).convert("RGB").save(cache_path)
                            images_saved += 1
                        except Exception as e:
                            print(f"  Warning: could not save image {image_id}: {e}")

                entry = {
                    "question_id": qid,
                    "question":    question,
                    "answer":      answer,
                    "imageId":     image_id,
                }
                if lang != "en" and qid in en_answers:
                    entry["answer_en"] = en_answers[qid]
                examples.append(entry)

        else:
            # Fallback: xgqa_repo (text only, no images)
            print(f"  [{lang}] not in floschne/xgqa, falling back to xgqa_repo...")
            repo_dir = "data/xgqa_repo"
            candidate_paths = [
                f"{repo_dir}/data/zero_shot/testdev_balanced_questions_{lang}.json",
                f"{repo_dir}/data/few_shot/{lang}/test.json",
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    raw = json.load(open(path, encoding="utf-8"))
                    items = list(raw.values()) if isinstance(raw, dict) else raw
                    for item in items[:200]:
                        examples.append({
                            "question_id": str(item.get("question_id", item.get("id", ""))),
                            "question":    item.get("question", ""),
                            "answer":      item.get("answer", item.get("full_answer", "")),
                            "imageId":     str(item.get("imageId", item.get("image_id", ""))),
                        })
                    print(f"  [{lang}] {len(examples)} examples from {path} (no images)")
                    break

        json.dump(examples, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        non_empty_q = sum(1 for e in examples if e["question"].strip())
        print(f"xGQA [{lang}]: {len(examples)} examples  "
              f"non-empty question={non_empty_q}  images saved={images_saved}  → {out_path}")

    print("\nxGQA download summary:")
    for lang in XGQA_LANGS:
        path = f"data/xgqa/xgqa_{lang}.json"
        if os.path.exists(path):
            print(f"  {lang}: {len(json.load(open(path)))} examples")
        else:
            print(f"  {lang}: missing")

    cached = len([f for f in os.listdir("data/gqa_images") if f.endswith(".jpg")])
    print(f"  gqa_images cache: {cached} images")
