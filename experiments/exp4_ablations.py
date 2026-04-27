import os
import json
import re
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

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
from models.constants import LANGUAGES, LAYER_RANGES

os.makedirs("outputs/ablations", exist_ok=True)

model, processor = load_model()
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]
examples = json.load(open("data/probing_set/probing_set.json"))

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
    sv = {}
    for L in layers_to_use:
        sv[L] = torch.stack(h_lang[L]).mean(0) - torch.stack(h_en[L]).mean(0)
    return sv


# ─── HELPER: evaluate MMMB for one lang/method/sv/layers ─────────────────────

def eval_mmmb(lang, method_name, sv, layers, use_blank=False):
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
            new_tok = generate_with_steering(model, inputs, sv, alpha=1.0, pivot_layers=layers)

        decoded = processor.decode(new_tok, skip_special_tokens=True).strip()
        match = re.search(r'\b([ABCD])\b', decoded.upper())
        predicted = match.group(1) if match else ""
        gold_answer = str(ex.get("answer", "")).upper()
        is_correct = predicted == gold_answer
        if is_correct:
            correct += 1
        total += 1

        print(f"\n[DBG MMMB] ex={ex_idx}  lang={lang}  method={method_name}  "
              f"{'blank' if use_blank else 'image'}")
        _dbg_image(img)
        print(f"  Q       : {ex['question']}")
        print(f"  Opts    : A={opts.get('A','')}  B={opts.get('B','')}  "
              f"C={opts.get('C','')}  D={opts.get('D','')}")
        print(f"  Model   : '{decoded}'  →  predicted='{predicted}'")
        print(f"  Gold    : '{gold_answer}'  |  {'✓ CORRECT' if is_correct else '✗ WRONG'}  "
              f"(running acc={correct}/{total}={correct/total:.3f})")

    return correct / total if total > 0 else 0.0


# ─── ABLATION 4a: Layer Range Ablation ───────────────────────────────────────

print("\n=== Ablation 4a: Layer Range Ablation ===")
layer_range_results = {}

all_ranges = dict(LAYER_RANGES)
all_ranges["ours"] = pivot_layers

# Pre-extract all 32 layers' answer-position residuals once per language.
# This replaces 4 ranges × N_examples × 2 forward passes with N_examples × 2,
# a 4× reduction. Memory cost: ~30 MB per language (60 × 32 × 4096 × float32).
ALL_LAYERS = list(range(32))
cached_residuals = {}  # {lang: {"lang": [{layer: array(4096,)}], "en": [...]}}

for lang in ["pt", "ru"]:
    mmmb_path = f"data/mmmb/mmmb_{lang}.json"
    if not os.path.exists(mmmb_path) or len(json.load(open(mmmb_path))) == 0:
        print(f"  Skipping [{lang}]: no MMMB data, skipping residual pre-extraction")
        continue
    print(f"  Pre-extracting all-layer residuals for [{lang}] (runs once for all ranges)...")
    res_lang_list = []
    res_en_list   = []
    for ex in examples:
        img = Image.open(ex["image_path"]).convert("RGB")
        inputs_lang, ans_pos    = get_image_text_inputs(processor, img, ex["questions"][lang])
        inputs_en,   ans_pos_en = get_image_text_inputs(processor, img, ex["questions"]["en"])
        res_lang = extract_residual_streams(model, inputs_lang, ALL_LAYERS)
        res_en   = extract_residual_streams(model, inputs_en,   ALL_LAYERS)
        res_lang_list.append({L: res_lang[L][ans_pos]    for L in ALL_LAYERS})
        res_en_list.append(  {L: res_en[L][ans_pos_en]   for L in ALL_LAYERS})
    cached_residuals[lang] = {"lang": res_lang_list, "en": res_en_list}
    print(f"  Done pre-extracting [{lang}]: {len(res_lang_list)} examples × 32 layers")

for range_name, layers_to_use in all_ranges.items():
    print(f"  Range: {range_name} = {layers_to_use}")
    layer_range_results[range_name] = {}
    for lang in ["pt", "ru"]:
        if lang not in cached_residuals:
            print(f"  Skipping {lang} (no MMMB data)")
            layer_range_results[range_name][f"{lang}_mmmb"] = None
            continue
        # Build steering vector from cached answer-position residuals
        sv = {}
        for L in layers_to_use:
            lang_vecs = torch.stack([torch.tensor(r[L]) for r in cached_residuals[lang]["lang"]])
            en_vecs   = torch.stack([torch.tensor(r[L]) for r in cached_residuals[lang]["en"]])
            sv[L] = lang_vecs.mean(0) - en_vecs.mean(0)
        acc = eval_mmmb(lang, f"{range_name}_steered", sv, layers_to_use)
        layer_range_results[range_name][f"{lang}_mmmb"] = acc
        print(f"  [{range_name}] {lang} MMMB accuracy: {acc:.3f}" if acc is not None else f"  [{range_name}] {lang} MMMB: N/A")

json.dump(layer_range_results, open("outputs/ablations/layer_range_ablation.json", "w"), indent=2)
print("Saved outputs/ablations/layer_range_ablation.json")

# ─── ABLATION 4b: Visual Token Ablation ──────────────────────────────────────

print("\n=== Ablation 4b: Visual Token Ablation ===")
visual_ablation = {"with_image": {}, "without_image": {}}

for lang in ["pt", "ru"]:
    mmmb_path = f"data/mmmb/mmmb_{lang}.json"
    if not os.path.exists(mmmb_path) or len(json.load(open(mmmb_path))) == 0:
        print(f"  Skipping {lang} (no MMMB data)")
        continue

    sv = torch.load(f"outputs/steering_vectors/steering_vec_{lang}.pt", map_location="cuda") \
         if os.path.exists(f"outputs/steering_vectors/steering_vec_{lang}.pt") else None

    # Load from Exp 3 results for with_image baseline
    baseline_file = f"outputs/eval_results/results_mmmb_{lang}_baseline.json"
    steered_file  = f"outputs/eval_results/results_mmmb_{lang}_steered_a1.0.json"

    if os.path.exists(baseline_file) and os.path.exists(steered_file):
        baseline_acc = json.load(open(baseline_file))["accuracy"]
        steered_acc  = json.load(open(steered_file))["accuracy"]
    else:
        baseline_acc = eval_mmmb(lang, "baseline", None, pivot_layers, use_blank=False)
        steered_acc  = eval_mmmb(lang, "steered", sv, pivot_layers, use_blank=False) if sv else None

    visual_ablation["with_image"][lang] = {
        "baseline": baseline_acc,
        "steered":  steered_acc,
    }

    # Blank image runs
    blank_baseline_acc = eval_mmmb(lang, "blank_baseline", None, pivot_layers, use_blank=True)
    blank_steered_acc  = eval_mmmb(lang, "blank_steered", sv, pivot_layers, use_blank=True) if sv else None

    visual_ablation["without_image"][lang] = {
        "baseline": blank_baseline_acc,
        "steered":  blank_steered_acc,
    }
    print(f"  [{lang}] with image: base={baseline_acc:.3f}, steered={steered_acc:.3f}" if baseline_acc is not None and steered_acc is not None else f"  [{lang}] with image: N/A")
    print(f"  [{lang}] blank:      base={blank_baseline_acc:.3f}, steered={blank_steered_acc:.3f}" if blank_baseline_acc is not None and blank_steered_acc is not None else f"  [{lang}] blank: N/A")

json.dump(visual_ablation, open("outputs/ablations/visual_token_ablation.json", "w"), indent=2)
print("Saved outputs/ablations/visual_token_ablation.json")

# ─── ABLATION 4c: α Sensitivity ──────────────────────────────────────────────

print("\n=== Ablation 4c: α Sensitivity ===")
alpha_sensitivity = {"mmmb": {}, "xgqa": {}}
alpha_values_str = ["0.0", "0.01", "0.02", "0.05", "0.07", "0.1", "0.2", "0.3", "0.4", "0.5"]

for dataset in ["mmmb", "xgqa"]:
    result_files = glob.glob(f"outputs/eval_results/results_{dataset}_*.json")
    for rf in result_files:
        data = json.load(open(rf))
        lang = data["lang"]
        method = data["method"]
        acc = data["accuracy"]

        if lang not in alpha_sensitivity[dataset]:
            alpha_sensitivity[dataset][lang] = {}

        if method == "baseline":
            alpha_sensitivity[dataset][lang]["0.0"] = acc
        elif method.startswith("steered_a"):
            alpha_str = method.replace("steered_a", "")
            alpha_sensitivity[dataset][lang][alpha_str] = acc

json.dump(alpha_sensitivity, open("outputs/ablations/alpha_sensitivity.json", "w"), indent=2)
print("Saved outputs/ablations/alpha_sensitivity.json")

# ─── ABLATION 4d: Cross-Lingual Transfer ─────────────────────────────────────

print("\n=== Ablation 4d: Cross-Lingual Transfer ===")
cross_lingual = {}

# Pairs: (source, target) — run on MMMB for pt and ru
pairs = [("pt", "ru"), ("ru", "pt"), ("de", "pt"), ("de", "ru")]

for src_lang, tgt_lang in pairs:
    sv_path = f"outputs/steering_vectors/steering_vec_{src_lang}.pt"
    mmmb_path = f"data/mmmb/mmmb_{tgt_lang}.json"
    if not os.path.exists(sv_path):
        print(f"  Skipping ({src_lang} → {tgt_lang}): no steering vector for {src_lang}")
        continue
    if not os.path.exists(mmmb_path) or len(json.load(open(mmmb_path))) == 0:
        print(f"  Skipping ({src_lang} → {tgt_lang}): no MMMB data for {tgt_lang}")
        continue
    sv = torch.load(sv_path, map_location="cuda")
    acc = eval_mmmb(tgt_lang, f"cross_{src_lang}_{tgt_lang}", sv, pivot_layers)
    key = f"vec_from_{src_lang}__eval_{tgt_lang}"
    cross_lingual[key] = acc
    print(f"  {key}: {acc:.3f}" if acc is not None else f"  {key}: N/A")

json.dump(cross_lingual, open("outputs/ablations/cross_lingual_transfer.json", "w"), indent=2)
print("Saved outputs/ablations/cross_lingual_transfer.json")

print("\nDONE: Experiment 4")
