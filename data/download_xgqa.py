import os
import json
import sys
import requests
from tqdm import tqdm
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

XGQA_LANGS = ["en", "zh", "bn"]   # ar not in adapter-hub/xGQA; bn and zh are


def parse_xgqa_example(ex):
    """Normalize one xGQA example to our standard format, handling multiple field name variants."""
    return {
        "question_id": str(ex.get("question_id", ex.get("questionId", ex.get("id", "")))),
        "question":    ex.get("question", ""),
        "answer":      ex.get("answer",   ex.get("full_answer", ex.get("label", ""))),
        "imageId":     str(ex.get("imageId", ex.get("image_id", ex.get("img_id", "")))),
    }


def download_gqa_image(image_id):
    """Download a GQA image by ID, caching to data/gqa_images/."""
    os.makedirs("data/gqa_images", exist_ok=True)
    cache_path = f"data/gqa_images/{image_id}.jpg"
    if os.path.exists(cache_path):
        return PIL.Image.open(cache_path).convert("RGB")
    url = f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
        return PIL.Image.open(cache_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: could not download GQA image {image_id}: {e}")
        return None


if __name__ == "__main__":
    os.makedirs("data/xgqa", exist_ok=True)
    os.makedirs("data/gqa_images", exist_ok=True)

    # STEP 1: Try HuggingFace first
    hf_success = False
    ds = None
    try:
        from datasets import load_dataset
        ds = load_dataset("adapter-hub/xGQA")
        hf_success = True
        print("Loaded xGQA from HuggingFace (adapter-hub/xGQA)")
    except Exception as e:
        print(f"HuggingFace xGQA failed: {e}")

    # STEP 2 (fallback): git clone the full xGQA repo, then parse JSONLs
    raw_data = {}
    if not hf_success:
        print("Falling back to GitHub xGQA clone...")
        import subprocess
        repo_dir = "data/xgqa_repo"
        if not os.path.exists(repo_dir):
            pat = os.environ.get("GITHUB_PAT", "")
            base_url = (f"https://{pat}@github.com/adapter-hub/xGQA"
                        if pat else "https://github.com/adapter-hub/xGQA")
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"
            for branch in ["main", "master"]:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch,
                     base_url, repo_dir],
                    capture_output=True, text=True, timeout=120, env=env
                )
                if result.returncode == 0:
                    print(f"  Cloned xGQA repo (branch={branch})")
                    break
                else:
                    print(f"  git clone branch={branch} failed: {result.stderr.strip()[:200]}")
            else:
                print("  Warning: git clone failed for both branches")

        for lang in XGQA_LANGS:
            raw_data[lang] = []
            # adapter-hub/xGQA uses flat files: data/few_shot/test_{lang}.json
            # and data/zero_shot/test_{lang}.json — try both, plus legacy paths
            candidate_paths = [
                f"{repo_dir}/data/few_shot/test_{lang}.json",
                f"{repo_dir}/data/zero_shot/test_{lang}.json",
                f"{repo_dir}/data/few_shot/test_{lang}.jsonl",
                f"{repo_dir}/data/zero_shot/test_{lang}.jsonl",
                f"{repo_dir}/data/few-shot/{lang}/test.jsonl",
                f"{repo_dir}/data/{lang}/test.jsonl",
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        content = f.read().strip()
                    # Handle both JSON array and JSONL formats
                    if content.startswith("["):
                        lines = json.loads(content)
                    else:
                        lines = [json.loads(l) for l in content.splitlines() if l.strip()]
                    raw_data[lang] = lines
                    print(f"  xGQA [{lang}]: {len(lines)} examples from {path}")
                    break
            else:
                print(f"  Warning: no xGQA data found for {lang} in cloned repo")

    # STEP 3: Save first 200 examples per language
    for lang in XGQA_LANGS:
        out_path = f"data/xgqa/xgqa_{lang}.json"

        if hf_success and ds is not None:
            try:
                if lang in ds:
                    split_data = list(ds[lang])
                elif "test" in ds:
                    split_data = list(ds["test"])
                else:
                    split_data = list(ds[list(ds.keys())[0]])
                examples = []
                for ex in split_data[:200]:
                    examples.append(parse_xgqa_example(dict(ex)))
            except Exception as e:
                print(f"  HF parse error for {lang}: {e}, using empty list")
                examples = []
        else:
            raw_examples = raw_data.get(lang, [])
            examples = []
            for ex in raw_examples[:200]:
                examples.append({
                    "question_id": str(ex.get("question_id", ex.get("id", ""))),
                    "question": ex.get("question", ""),
                    "answer": ex.get("answer", ex.get("full_answer", "")),
                    "imageId": str(ex.get("imageId", ex.get("image_id", ""))),
                })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"xGQA [{lang}]: {len(examples)} examples → {out_path}")

    # STEP 4: Print summary
    print("\nxGQA download summary:")
    for lang in XGQA_LANGS:
        path = f"data/xgqa/xgqa_{lang}.json"
        if os.path.exists(path):
            n = len(json.load(open(path)))
            print(f"  {lang}: {n} examples")
        else:
            print(f"  {lang}: missing")
