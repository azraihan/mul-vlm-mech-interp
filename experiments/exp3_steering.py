import os
import json
import re
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sacrebleu.metrics import CHRF
from bert_score import BERTScorer

try:
    from IPython.display import display as _ipy_display
    _ipy_ok = True
except ImportError:
    _ipy_ok = False

def _dbg_image(img):
    if _ipy_ok:
        _ipy_display(img)
    else:
        print("  [IMG] (IPython not available — save image to view)")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import (load_model, get_image_text_inputs,
                                   extract_residual_streams, generate_with_steering)
from models.constants import LANGUAGES, LANG_NAMES

os.makedirs("outputs/steering_vectors", exist_ok=True)
os.makedirs("outputs/eval_results", exist_ok=True)

model, processor = load_model()
examples = json.load(open("data/probing_set/probing_set.json"))
pivot_data = json.load(open("outputs/pivot_layers.json"))
pivot_layers = pivot_data["pivot_layers"]

# ─── PART A: Extract steering vectors ─────────────────────────────────────────

for lang in LANGUAGES:
    h_lang = {L: [] for L in pivot_layers}
    h_en   = {L: [] for L in pivot_layers}

    for ex in tqdm(examples, desc=f"Steering vector [{lang}]"):
        img = Image.open(ex["image_path"]).convert("RGB")
        inputs_lang, ans_pos    = get_image_text_inputs(processor, img, ex["questions"][lang])
        inputs_en,   ans_pos_en = get_image_text_inputs(processor, img, ex["questions"]["en"])

        res_lang = extract_residual_streams(model, inputs_lang, pivot_layers)
        res_en   = extract_residual_streams(model, inputs_en,   pivot_layers)

        for L in pivot_layers:
            h_lang[L].append(torch.tensor(res_lang[L][ans_pos]))     # (4096,)
            h_en[L].append(  torch.tensor(res_en[L][ans_pos_en]))

    steering_vec = {}
    for L in pivot_layers:
        stacked_lang = torch.stack(h_lang[L])   # (N, 4096)
        stacked_en   = torch.stack(h_en[L])     # (N, 4096)
        v = stacked_lang.mean(0) - stacked_en.mean(0)  # (4096,)
        steering_vec[L] = v

        # Stability check
        n = len(h_lang[L])
        if n >= 2:
            odd  = torch.stack(h_lang[L][0::2]).mean(0) - torch.stack(h_en[L][0::2]).mean(0)
            even = torch.stack(h_lang[L][1::2]).mean(0) - torch.stack(h_en[L][1::2]).mean(0)
            cos  = torch.nn.functional.cosine_similarity(
                odd.unsqueeze(0), even.unsqueeze(0)
            ).item()
            status = "OK" if cos >= 0.85 else "WARN (unstable — consider more examples)"
            print(f"  [{lang}] layer {L}: cosine stability = {cos:.3f}  {status}")

    torch.save(steering_vec, f"outputs/steering_vectors/steering_vec_{lang}.pt")
    print(f"Saved steering vector for [{lang}]")

print("DONE: Steering vector extraction")

# ─── PART B: Evaluation ───────────────────────────────────────────────────────

import PIL.Image as PILImage

MMMB_LANGS = [lang for lang in ["en", "pt", "ru"]
               if os.path.exists(f"data/mmmb/mmmb_{lang}.json") and
               len(json.load(open(f"data/mmmb/mmmb_{lang}.json"))) > 0]

print(f"\nEvaluating MMMB for langs: {MMMB_LANGS}")

for lang in MMMB_LANGS:
    mmmb = json.load(open(f"data/mmmb/mmmb_{lang}.json"))[:200]
    sv_lang = lang if lang in LANGUAGES else None   # "en" has no steering vector

    methods = ["baseline"] + [f"steered_a{a}" for a in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]]

    for method in methods:
        if method != "baseline" and sv_lang is None:
            continue

        if method != "baseline" and sv_lang is not None:
            sv = torch.load(f"outputs/steering_vectors/steering_vec_{sv_lang}.pt", map_location="cuda")
        else:
            sv = None

        correct = 0
        total = 0
        for ex_idx, ex in enumerate(tqdm(mmmb, desc=f"MMMB {lang} {method}")):
            # Load image
            try:
                if isinstance(ex["image"], str) and ex["image"] and os.path.exists(ex["image"]):
                    img = PILImage.open(ex["image"]).convert("RGB")
                elif isinstance(ex["image"], str) and ex["image"].startswith("data:image"):
                    import base64, io as io_mod
                    b64 = ex["image"].split(",", 1)[1]
                    img = PILImage.open(io_mod.BytesIO(base64.b64decode(b64))).convert("RGB")
                else:
                    # fallback blank image
                    img = PILImage.new("RGB", (336, 336), color=(255, 255, 255))
            except Exception:
                img = PILImage.new("RGB", (336, 336), color=(255, 255, 255))

            opts = ex.get("options", {})
            q = (ex["question"] +
                 "\nOptions:\nA. " + opts.get("A", "") +
                 "\nB. " + opts.get("B", "") +
                 "\nC. " + opts.get("C", "") +
                 "\nD. " + opts.get("D", "") +
                 "\nAnswer with only the correct option letter (A, B, C, or D).")
            inputs, _ = get_image_text_inputs(processor, img, q)

            if method == "baseline":
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                new_tok = out[0][inputs["input_ids"].shape[1]:]
            else:
                alpha = float(method.split("a")[1])
                new_tok = generate_with_steering(model, inputs, sv, alpha, pivot_layers)

            decoded = processor.decode(new_tok, skip_special_tokens=True).strip()
            match = re.search(r'\b([ABCD])\b', decoded.upper())
            predicted = match.group(1) if match else ""
            gold_answer = str(ex.get("answer", "")).upper()
            is_correct = predicted == gold_answer
            if is_correct:
                correct += 1
            total += 1

            print(f"\n[DBG MMMB] ex={ex_idx}  lang={lang}  method={method}")
            _dbg_image(img)
            print(f"  Q       : {q}")
            print(f"  Model   : '{decoded}'  →  predicted='{predicted}'")
            print(f"  Gold    : '{gold_answer}'  |  {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        acc = correct / total if total > 0 else 0.0
        result = {
            "accuracy": acc, "correct": correct, "total": total,
            "method": method, "lang": lang, "dataset": "mmmb"
        }
        fname = f"outputs/eval_results/results_mmmb_{lang}_{method}.json"
        json.dump(result, open(fname, "w"), indent=2)
        print(f"  {fname}: accuracy={acc:.3f}")

# ─── xGQA metric helpers (preloaded once) ────────────────────────────────────

_ARTICLES = {
    "en": re.compile(r"^(a|an|the)\s+"),
    "pt": re.compile(r"^(o|a|os|as|um|uma|uns|umas)\s+"),
    "de": re.compile(r"^(der|die|das|des|dem|den|ein|eine|einer|einem|einen|eines)\s+"),
    "ru": None,  # Russian has no articles
}

def _normalize(text, lang="en"):
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    pat = _ARTICLES.get(lang)
    if pat:
        text = pat.sub("", text)
    return text.strip()

_chrf = CHRF()

print("[xGQA] Loading BERTScorer (bert-base-multilingual-cased)...")
_bert_scorer = BERTScorer(
    model_type="bert-base-multilingual-cased",
    num_layers=9,
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("[xGQA] BERTScorer ready.")

# ─── xGQA evaluation ──────────────────────────────────────────────────────────

XGQA_LANGS_EVAL = [lang for lang in ["en", "de", "pt", "ru"]
                   if os.path.exists(f"data/xgqa/xgqa_{lang}.json") and
                   len(json.load(open(f"data/xgqa/xgqa_{lang}.json"))) > 0]

print(f"\nEvaluating xGQA for langs: {XGQA_LANGS_EVAL}")

from data.download_xgqa import download_gqa_image

# Pre-check: attempt one image download to detect dead hosting
_test_img = download_gqa_image(XGQA_LANGS_EVAL and
    json.load(open(f"data/xgqa/xgqa_{XGQA_LANGS_EVAL[0]}.json"))[0]["imageId"])
_images_available = _test_img is not None
if not _images_available:
    print("[WARN] xGQA image hosting (VG_100K) is unreachable — all images will be "
          "blank white fallbacks. xGQA accuracy numbers are NOT valid and should be "
          "excluded from the paper.")
else:
    print("[OK] xGQA image hosting reachable.")

for lang in XGQA_LANGS_EVAL:
    xgqa = json.load(open(f"data/xgqa/xgqa_{lang}.json"))[:200]
    sv_lang = lang if lang in LANGUAGES else None

    methods = ["baseline"] + [f"steered_a{a}" for a in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]]

    for method in methods:
        if method != "baseline" and sv_lang is None:
            continue

        if method != "baseline" and sv_lang is not None:
            sv = torch.load(f"outputs/steering_vectors/steering_vec_{sv_lang}.pt", map_location="cuda")
        else:
            sv = None

        em_sum = 0
        chrf_sum = 0.0
        bertscore_sum = 0.0
        total = 0
        img_failures = 0
        empty_questions = 0
        for ex_idx, ex in enumerate(tqdm(xgqa, desc=f"xGQA {lang} {method}")):
            image_id = ex.get("imageId", "")
            img = download_gqa_image(image_id)
            if img is None:
                img_failures += 1
                img = PILImage.new("RGB", (336, 336), color=(255, 255, 255))

            question = ex.get("question", "")
            if not question.strip():
                empty_questions += 1
            question = question + f"\nAnswer in a single word only, in {LANG_NAMES.get(lang, lang)}."

            inputs, _ = get_image_text_inputs(processor, img, question)

            if method == "baseline":
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                new_tok = out[0][inputs["input_ids"].shape[1]:]
            else:
                alpha = float(method.split("a")[1])
                new_tok = generate_with_steering(model, inputs, sv, alpha, pivot_layers,
                                                 max_new_tokens=10)

            decoded = processor.decode(new_tok, skip_special_tokens=True).strip()
            pred_raw = decoded.lower().strip().split()[0] if decoded.strip() else ""
            gold_raw = str(ex.get("answer", ""))
            gold_en_raw = str(ex.get("answer_en", ""))

            pred_norm    = _normalize(pred_raw, lang)
            gold_norm    = _normalize(gold_raw, lang)
            gold_en_norm = _normalize(gold_en_raw, "en")

            # Normalized exact match (accept target-lang or English gold)
            em = int(pred_norm == gold_norm or (gold_en_norm and pred_norm == gold_en_norm))
            em_sum += em

            # chrF (0–100 → 0–1); use target-lang gold
            ex_chrf = _chrf.sentence_score(pred_norm, [gold_norm]).score / 100.0
            chrf_sum += ex_chrf

            # BERTScore F1 with preloaded multilingual model
            ref = gold_norm if gold_norm else gold_en_norm
            ex_bs = 0.0
            if pred_norm and ref:
                _, _, F1 = _bert_scorer.score([pred_norm], [ref])
                ex_bs = F1[0].item()
            bertscore_sum += ex_bs

            total += 1

            print(f"\n[DBG xGQA] ex={ex_idx}  lang={lang}  method={method}")
            _dbg_image(img)
            print(f"  Q       : {ex.get('question', '')}")
            print(f"  Model   : '{decoded}'  →  predicted='{pred_norm}'")
            print(f"  Gold    : '{gold_norm}'" + (f"  (en: '{gold_en_norm}')" if gold_en_norm else ""))
            print(f"  EM={em}  chrF={ex_chrf:.3f}  BERTScore={ex_bs:.3f}")

        if img_failures > 0:
            print(f"  [WARN] xGQA {lang} {method}: {img_failures}/{total} images failed to download — used blank fallback")
        if empty_questions > 0:
            print(f"  [WARN] xGQA {lang} {method}: {empty_questions}/{total} examples had empty questions")
        if img_failures == 0 and empty_questions == 0:
            print(f"  [OK] xGQA {lang} {method}: all {total} images and questions present")

        n = total or 1
        exact_match  = em_sum / n
        chrf_avg     = chrf_sum / n
        bertscore_f1 = bertscore_sum / n
        result = {
            "accuracy":     exact_match,   # kept for exp4 compatibility
            "exact_match":  exact_match,
            "chrf":         chrf_avg,
            "bertscore_f1": bertscore_f1,
            "total": total,
            "img_failures": img_failures, "empty_questions": empty_questions,
            "images_valid": _images_available,
            "method": method, "lang": lang, "dataset": "xgqa"
        }
        fname = f"outputs/eval_results/results_xgqa_{lang}_{method}.json"
        json.dump(result, open(fname, "w"), indent=2)
        print(f"  {fname}: EM={exact_match:.3f}  chrF={chrf_avg:.3f}  BERTScore={bertscore_f1:.3f}")

print("DONE: Experiment 3")
