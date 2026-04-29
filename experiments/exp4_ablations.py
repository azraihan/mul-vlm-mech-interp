import gc
import os
import json
import re
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
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
from models.llava_wrapper import (load_model, get_image_text_inputs, get_blank_image_inputs,
                                   extract_residual_streams, generate_with_steering)
from models.constants import LANGUAGES, LANG_NAMES, LAYER_RANGES

os.makedirs("outputs/ablations", exist_ok=True)

model, processor = load_model()
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]
examples = json.load(open("data/probing_set/probing_set.json"))
print(f"[EXP4] Using {len(examples)} examples from probing set")

# Alpha sweep — matches Exp 3
ALPHA_VALUES = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]

XGQA_LANGS = [lang for lang in ["en", "de", "pt", "ru"]
              if os.path.exists(f"data/xgqa/xgqa_{lang}.json") and
              len(json.load(open(f"data/xgqa/xgqa_{lang}.json"))) > 0]

# Initialise xGQA metrics once — BERTScorer is expensive to load repeatedly
_chrf = CHRF()
if XGQA_LANGS:
    print("[xGQA] Loading BERTScorer (bert-base-multilingual-cased)...")
    _bert_scorer = BERTScorer(
        model_type="bert-base-multilingual-cased",
        num_layers=9,
        rescale_with_baseline=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[xGQA] BERTScorer ready.")
else:
    _bert_scorer = None

# Pre-check xGQA image hosting once
_images_available = False
if XGQA_LANGS:
    from data.download_xgqa import download_gqa_image as _dgqa
    _first_id = json.load(open(f"data/xgqa/xgqa_{XGQA_LANGS[0]}.json"))[0].get("imageId", "")
    _test_img = _dgqa(_first_id)
    _images_available = _test_img is not None
    if not _images_available:
        print("[WARN] xGQA image hosting unreachable — blank fallbacks will be used; "
              "xGQA numbers are NOT valid.")
    else:
        print("[OK] xGQA image hosting reachable.")

_ARTICLES = {
    "en": re.compile(r"^(a|an|the)\s+"),
    "pt": re.compile(r"^(o|a|os|as|um|uma|uns|umas)\s+"),
    "de": re.compile(r"^(der|die|das|des|dem|den|ein|eine|einer|einem|einen|eines)\s+"),
    "ru": None,
}

def _normalize(text, lang="en"):
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    pat = _ARTICLES.get(lang)
    if pat:
        text = pat.sub("", text)
    return text.strip()


# ─── HELPER: extract steering vector for a given set of layers ───────────────

def extract_steering_vec(lang, layers_to_use):
    h_lang = {L: [] for L in layers_to_use}
    h_en   = {L: [] for L in layers_to_use}
    for ex in examples:
        img = Image.open(ex["image_path"]).convert("RGB")
        inputs_lang, ans_pos    = get_image_text_inputs(processor, img, ex["questions"][lang])
        inputs_en,   ans_pos_en = get_image_text_inputs(processor, img, ex["questions"]["en"])
        res_lang = extract_residual_streams(model, inputs_lang, layers_to_use)
        res_en   = extract_residual_streams(model, inputs_en,   layers_to_use)
        for L in layers_to_use:
            h_lang[L].append(torch.tensor(res_lang[L][ans_pos]))
            h_en[L].append(  torch.tensor(res_en[L][ans_pos_en]))
        del res_lang, res_en
    sv = {}
    for L in layers_to_use:
        sv[L] = torch.stack(h_lang[L]).mean(0) - torch.stack(h_en[L]).mean(0)
    return sv


# ─── HELPER: evaluate MMMB for one lang/method/sv/layers/alpha ───────────────

def eval_mmmb(lang, method_name, sv, layers, alpha=1.0, use_blank=False):
    mmmb_path = f"data/mmmb/mmmb_{lang}.json"
    if not os.path.exists(mmmb_path):
        return None
    mmmb = json.load(open(mmmb_path))[:200]
    if len(mmmb) == 0:
        return None

    correct = 0
    total = 0
    for ex_idx, ex in enumerate(mmmb):
        try:
            if use_blank:
                img = Image.new("RGB", (336, 336), color=(255, 255, 255))
            elif isinstance(ex["image"], str) and ex["image"] and os.path.exists(ex["image"]):
                img = Image.open(ex["image"]).convert("RGB")
            else:
                img = Image.new("RGB", (336, 336), color=(255, 255, 255))
        except Exception:
            img = Image.new("RGB", (336, 336), color=(255, 255, 255))

        opts = ex.get("options", {})
        q = (ex["question"] +
             "\nOptions:\nA. " + opts.get("A", "") +
             "\nB. " + opts.get("B", "") +
             "\nC. " + opts.get("C", "") +
             "\nD. " + opts.get("D", "") +
             "\nAnswer with only the correct option letter (A, B, C, or D).")

        if use_blank:
            inputs, _ = get_blank_image_inputs(processor, q)
        else:
            inputs, _ = get_image_text_inputs(processor, img, q)

        if sv is None:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            new_tok = out[0][inputs["input_ids"].shape[1]:]
        else:
            new_tok = generate_with_steering(model, inputs, sv, alpha=alpha, pivot_layers=layers)

        decoded = processor.decode(new_tok, skip_special_tokens=True).strip()
        match = re.search(r'\b([ABCD])\b', decoded.upper())
        predicted = match.group(1) if match else ""
        gold_answer = str(ex.get("answer", "")).upper()
        if predicted == gold_answer:
            correct += 1
        total += 1

        print(f"\n[DBG MMMB] ex={ex_idx}  lang={lang}  method={method_name}  "
              f"{'blank' if use_blank else 'image'}")
        _dbg_image(img)
        print(f"  Q       : {ex['question']}")
        print(f"  Opts    : A={opts.get('A','')}  B={opts.get('B','')}  "
              f"C={opts.get('C','')}  D={opts.get('D','')}")
        print(f"  Model   : '{decoded}'  →  predicted='{predicted}'")
        print(f"  Gold    : '{gold_answer}'  |  {'✓' if predicted == gold_answer else '✗'}  "
              f"(running acc={correct}/{total}={correct/total:.3f})")

    return correct / total if total > 0 else 0.0


# ─── HELPER: evaluate xGQA for one lang/method/sv/layers/alpha ───────────────

def eval_xgqa(lang, method_name, sv, layers, alpha=1.0, use_blank=False):
    xgqa_path = f"data/xgqa/xgqa_{lang}.json"
    if not os.path.exists(xgqa_path):
        return None
    xgqa = json.load(open(xgqa_path))[:200]
    if not xgqa:
        return None

    from data.download_xgqa import download_gqa_image

    em_sum       = 0
    chrf_sum     = 0.0
    bertscore_sum = 0.0
    total        = 0
    img_failures = 0
    empty_questions = 0

    for ex_idx, ex in enumerate(tqdm(xgqa, desc=f"xGQA {lang} {method_name}")):
        if use_blank:
            img = Image.new("RGB", (336, 336), color=(255, 255, 255))
        else:
            image_id = ex.get("imageId", "")
            img = download_gqa_image(image_id)
            if img is None:
                img_failures += 1
                img = Image.new("RGB", (336, 336), color=(255, 255, 255))

        question = ex.get("question", "")
        if not question.strip():
            empty_questions += 1
        question = question + f"\nAnswer in a single word only, in {LANG_NAMES.get(lang, lang)}."

        if use_blank:
            inputs, _ = get_blank_image_inputs(processor, question)
        else:
            inputs, _ = get_image_text_inputs(processor, img, question)

        if sv is None:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            new_tok = out[0][inputs["input_ids"].shape[1]:]
        else:
            new_tok = generate_with_steering(model, inputs, sv, alpha=alpha,
                                             pivot_layers=layers, max_new_tokens=10)

        decoded = processor.decode(new_tok, skip_special_tokens=True).strip()
        pred_raw    = decoded.lower().strip().split()[0] if decoded.strip() else ""
        gold_raw    = str(ex.get("answer", ""))
        gold_en_raw = str(ex.get("answer_en", ""))

        pred_norm    = _normalize(pred_raw, lang)
        gold_norm    = _normalize(gold_raw, lang)
        gold_en_norm = _normalize(gold_en_raw, "en")

        em = int(pred_norm == gold_norm or (bool(gold_en_norm) and pred_norm == gold_en_norm))
        em_sum += em

        ex_chrf = _chrf.sentence_score(pred_norm, [gold_norm]).score / 100.0
        chrf_sum += ex_chrf

        ref = gold_norm if gold_norm else gold_en_norm
        ex_bs = 0.0
        if pred_norm and ref and _bert_scorer is not None:
            _, _, F1 = _bert_scorer.score([pred_norm], [ref])
            ex_bs = F1[0].item()
        bertscore_sum += ex_bs

        total += 1

        print(f"\n[DBG xGQA] ex={ex_idx}  lang={lang}  method={method_name}  "
              f"{'blank' if use_blank else 'image'}")
        _dbg_image(img)
        print(f"  Q       : {ex.get('question', '')}")
        print(f"  Model   : '{decoded}'  →  predicted='{pred_norm}'")
        print(f"  Gold    : '{gold_norm}'" + (f"  (en: '{gold_en_norm}')" if gold_en_norm else ""))
        print(f"  EM={em}  chrF={ex_chrf:.3f}  BERTScore={ex_bs:.3f}")

    if img_failures > 0:
        print(f"  [WARN] xGQA {lang} {method_name}: {img_failures}/{total} images failed — used blank fallback")
    if empty_questions > 0:
        print(f"  [WARN] xGQA {lang} {method_name}: {empty_questions}/{total} examples had empty questions")
    if img_failures == 0 and empty_questions == 0:
        print(f"  [OK] xGQA {lang} {method_name}: all {total} images and questions present")

    n = total or 1
    return {
        "exact_match":  em_sum / n,
        "chrf":         chrf_sum / n,
        "bertscore_f1": bertscore_sum / n,
        "total":        total,
        "img_failures": img_failures,
        "empty_questions": empty_questions,
        "images_valid": _images_available,
    }


# ─── ABLATION 4a: Layer Range Ablation ───────────────────────────────────────

print("\n=== Ablation 4a: Layer Range Ablation ===")

if os.path.exists("outputs/ablations/layer_range_ablation.json"):
    print("[SKIP] layer_range_ablation.json already exists — skipping.")
    layer_range_results = json.load(open("outputs/ablations/layer_range_ablation.json"))
else:
    layer_range_results = {}
    all_ranges = dict(LAYER_RANGES)
    all_ranges["ours"] = pivot_layers

    for range_name, layers_to_use in all_ranges.items():
        print(f"\n  Range: {range_name} = {layers_to_use}")
        layer_range_results[range_name] = {}

        # ── MMMB ──
        for lang in ["pt", "ru"]:
            if not (os.path.exists(f"data/mmmb/mmmb_{lang}.json") and
                    len(json.load(open(f"data/mmmb/mmmb_{lang}.json"))) > 0):
                print(f"  Skipping {lang} MMMB (no data)")
                layer_range_results[range_name][f"{lang}_mmmb"] = None
                continue
            print(f"  [{range_name}] Extracting steering vec for {lang} MMMB...")
            sv = extract_steering_vec(lang, layers_to_use)
            alpha_results = {}
            alpha_results["0.0"] = eval_mmmb(lang, f"{range_name}_baseline", None, layers_to_use)
            for alpha in ALPHA_VALUES:
                acc = eval_mmmb(lang, f"{range_name}_a{alpha}", sv, layers_to_use, alpha=alpha)
                alpha_results[str(alpha)] = acc
                print(f"  [{range_name}] {lang} MMMB α={alpha}: " +
                      (f"{acc:.3f}" if acc is not None else "N/A"))
            layer_range_results[range_name][f"{lang}_mmmb"] = alpha_results
            del sv
            gc.collect()
            torch.cuda.empty_cache()

        # ── xGQA ──
        for lang in [l for l in XGQA_LANGS if l in LANGUAGES]:
            alpha_results = {}
            print(f"  [{range_name}] Extracting steering vec for {lang} xGQA...")
            sv = extract_steering_vec(lang, layers_to_use)
            alpha_results["0.0"] = eval_xgqa(lang, f"{range_name}_baseline", None, layers_to_use)
            for alpha in ALPHA_VALUES:
                acc = eval_xgqa(lang, f"{range_name}_a{alpha}", sv, layers_to_use, alpha=alpha)
                alpha_results[str(alpha)] = acc
                em = acc["exact_match"] if isinstance(acc, dict) else acc
                print(f"  [{range_name}] {lang} xGQA α={alpha}: " +
                      (f"EM={em:.3f}" if em is not None else "N/A"))
            layer_range_results[range_name][f"{lang}_xgqa"] = alpha_results
            del sv
            gc.collect()
            torch.cuda.empty_cache()

    json.dump(layer_range_results, open("outputs/ablations/layer_range_ablation.json", "w"), indent=2)
    print("Saved outputs/ablations/layer_range_ablation.json")


# ─── ABLATION 4b: Visual Token Ablation ──────────────────────────────────────

print("\n=== Ablation 4b: Visual Token Ablation ===")

if os.path.exists("outputs/ablations/visual_token_ablation.json"):
    print("[SKIP] visual_token_ablation.json already exists — skipping.")
    visual_ablation = json.load(open("outputs/ablations/visual_token_ablation.json"))
else:
    visual_ablation = {"with_image": {}, "without_image": {}}

    def _load_or_run_mmmb(lang, alpha, sv, use_blank=False):
        """Load from Exp 3 result file when possible; otherwise run fresh."""
        if not use_blank:
            method = "baseline" if alpha == 0.0 else f"steered_a{alpha}"
            path = f"outputs/eval_results/results_mmmb_{lang}_{method}.json"
            if os.path.exists(path):
                return json.load(open(path))["accuracy"]
        if alpha == 0.0:
            return eval_mmmb(lang, "baseline", None, pivot_layers, use_blank=use_blank)
        return eval_mmmb(lang, f"steered_a{alpha}", sv, pivot_layers,
                         alpha=alpha, use_blank=use_blank) if sv is not None else None

    def _load_or_run_xgqa(lang, alpha, sv, use_blank=False):
        """Load from Exp 3 result file when possible; otherwise run fresh."""
        if not use_blank:
            method = "baseline" if alpha == 0.0 else f"steered_a{alpha}"
            path = f"outputs/eval_results/results_xgqa_{lang}_{method}.json"
            if os.path.exists(path):
                d = json.load(open(path))
                return {
                    "exact_match":  d.get("exact_match", d.get("accuracy")),
                    "chrf":         d.get("chrf"),
                    "bertscore_f1": d.get("bertscore_f1"),
                    "total":        d.get("total"),
                    "img_failures": d.get("img_failures", 0),
                    "empty_questions": d.get("empty_questions", 0),
                    "images_valid": d.get("images_valid", _images_available),
                }
        if alpha == 0.0:
            return eval_xgqa(lang, "baseline", None, pivot_layers, use_blank=use_blank)
        return eval_xgqa(lang, f"steered_a{alpha}", sv, pivot_layers,
                         alpha=alpha, use_blank=use_blank) if sv is not None else None

    # ── MMMB ──
    for lang in ["pt", "ru"]:
        if not os.path.exists(f"data/mmmb/mmmb_{lang}.json") or \
                len(json.load(open(f"data/mmmb/mmmb_{lang}.json"))) == 0:
            print(f"  Skipping {lang} MMMB (no data)")
            continue

        sv = (torch.load(f"outputs/steering_vectors/steering_vec_{lang}.pt", map_location="cuda")
              if os.path.exists(f"outputs/steering_vectors/steering_vec_{lang}.pt") else None)

        with_img, without_img = {}, {}
        for alpha in [0.0] + ALPHA_VALUES:
            with_img[str(alpha)]    = _load_or_run_mmmb(lang, alpha, sv, use_blank=False)
            without_img[str(alpha)] = _load_or_run_mmmb(lang, alpha, sv, use_blank=True)

        visual_ablation["with_image"][f"{lang}_mmmb"]    = with_img
        visual_ablation["without_image"][f"{lang}_mmmb"] = without_img
        print(f"  [{lang}] MMMB done.")

    # ── xGQA ──
    for lang in [l for l in XGQA_LANGS if l in LANGUAGES]:
        sv_lang = lang
        sv = (torch.load(f"outputs/steering_vectors/steering_vec_{sv_lang}.pt", map_location="cuda")
              if sv_lang and os.path.exists(f"outputs/steering_vectors/steering_vec_{sv_lang}.pt")
              else None)

        with_img, without_img = {}, {}
        for alpha in [0.0] + ALPHA_VALUES:
            with_img[str(alpha)]    = _load_or_run_xgqa(lang, alpha, sv, use_blank=False)
            without_img[str(alpha)] = _load_or_run_xgqa(lang, alpha, sv, use_blank=True)

        visual_ablation["with_image"][f"{lang}_xgqa"]    = with_img
        visual_ablation["without_image"][f"{lang}_xgqa"] = without_img
        print(f"  [{lang}] xGQA done.")

    json.dump(visual_ablation, open("outputs/ablations/visual_token_ablation.json", "w"), indent=2)
    print("Saved outputs/ablations/visual_token_ablation.json")


# ─── ABLATION 4c: α Sensitivity ──────────────────────────────────────────────

print("\n=== Ablation 4c: α Sensitivity ===")

if os.path.exists("outputs/ablations/alpha_sensitivity.json"):
    print("[SKIP] alpha_sensitivity.json already exists — skipping.")
else:
    alpha_sensitivity = {"mmmb": {}, "xgqa": {}}

    for dataset in ["mmmb", "xgqa"]:
        for rf in glob.glob(f"outputs/eval_results/results_{dataset}_*.json"):
            data = json.load(open(rf))
            lang   = data["lang"]
            method = data["method"]
            acc    = data["accuracy"]
            if lang not in alpha_sensitivity[dataset]:
                alpha_sensitivity[dataset][lang] = {}
            if method == "baseline":
                alpha_sensitivity[dataset][lang]["0.0"] = acc
            elif method.startswith("steered_a"):
                alpha_sensitivity[dataset][lang][method.replace("steered_a", "")] = acc

    json.dump(alpha_sensitivity, open("outputs/ablations/alpha_sensitivity.json", "w"), indent=2)
    print("Saved outputs/ablations/alpha_sensitivity.json")


# ─── ABLATION 4d: Cross-Lingual Transfer ─────────────────────────────────────

print("\n=== Ablation 4d: Cross-Lingual Transfer ===")

if os.path.exists("outputs/ablations/cross_lingual_transfer.json"):
    print("[SKIP] cross_lingual_transfer.json already exists — skipping.")
else:
    from itertools import permutations
    cross_lingual = {}
    pairs = [(s, t) for s, t in permutations(LANGUAGES, 2)]

    for src_lang, tgt_lang in pairs:
        sv_path = f"outputs/steering_vectors/steering_vec_{src_lang}.pt"
        if not os.path.exists(sv_path):
            print(f"  Skipping ({src_lang} → {tgt_lang}): no steering vector for {src_lang}")
            continue
        sv = torch.load(sv_path, map_location="cuda")

        # ── MMMB ──
        mmmb_path = f"data/mmmb/mmmb_{tgt_lang}.json"
        if os.path.exists(mmmb_path) and len(json.load(open(mmmb_path))) > 0:
            mmmb_alpha_results = {}
            mmmb_alpha_results["0.0"] = eval_mmmb(
                tgt_lang, f"cross_{src_lang}_{tgt_lang}_baseline", None, pivot_layers)
            for alpha in ALPHA_VALUES:
                acc = eval_mmmb(
                    tgt_lang, f"cross_{src_lang}_{tgt_lang}_a{alpha}", sv, pivot_layers, alpha=alpha)
                mmmb_alpha_results[str(alpha)] = acc
                print(f"  [{src_lang}→{tgt_lang}] MMMB α={alpha}: "
                      + (f"{acc:.3f}" if acc is not None else "N/A"))
            cross_lingual[f"vec_from_{src_lang}__eval_{tgt_lang}_mmmb"] = mmmb_alpha_results
        else:
            print(f"  Skipping ({src_lang} → {tgt_lang}) MMMB: no data for {tgt_lang}")

        # ── xGQA ──
        if tgt_lang in XGQA_LANGS:
            xgqa_alpha_results = {}
            xgqa_alpha_results["0.0"] = eval_xgqa(
                tgt_lang, f"cross_{src_lang}_{tgt_lang}_baseline", None, pivot_layers)
            for alpha in ALPHA_VALUES:
                acc = eval_xgqa(
                    tgt_lang, f"cross_{src_lang}_{tgt_lang}_a{alpha}", sv, pivot_layers, alpha=alpha)
                xgqa_alpha_results[str(alpha)] = acc
                em = acc["exact_match"] if isinstance(acc, dict) else acc
                print(f"  [{src_lang}→{tgt_lang}] xGQA α={alpha}: "
                      + (f"EM={em:.3f}" if em is not None else "N/A"))
            cross_lingual[f"vec_from_{src_lang}__eval_{tgt_lang}_xgqa"] = xgqa_alpha_results
        else:
            print(f"  Skipping ({src_lang} → {tgt_lang}) xGQA: no data for {tgt_lang}")

    json.dump(cross_lingual, open("outputs/ablations/cross_lingual_transfer.json", "w"), indent=2)
    print("Saved outputs/ablations/cross_lingual_transfer.json")

print("\nDONE: Experiment 4")
