import os
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import (load_model, get_image_text_inputs, get_blank_image_inputs,
                                   extract_residual_streams, apply_logit_lens,
                                   compute_pmi_baseline)
from models.constants import LANGUAGES, NUM_LAYERS

os.makedirs("outputs/logit_lens", exist_ok=True)
model, processor = load_model()
baseline_log_probs = compute_pmi_baseline(model)   # (32000,) — model's prior log P(token)
examples = json.load(open("data/probing_set/probing_set.json"))
N = len(examples)

en_probs = np.full((2, 4, N, NUM_LAYERS), np.nan, dtype=np.float32)
tl_probs = np.full((2, 4, N, NUM_LAYERS), np.nan, dtype=np.float32)

# ── Tokenization diagnostic ───────────────────────────────────────────────────
print("\n[TOK] ── Tokenization diagnostic (all categories × all languages) ──")
prefix = "ASSISTANT:"
prefix_ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
seen_categories = {}
for ex in examples:
    cat = ex.get("category")
    if cat in seen_categories:
        continue
    seen_categories[cat] = True
    for lang in ["en", "fr", "ar", "zh", "bn"]:
        word = ex["questions"].get(lang, "")
        # re-tokenize the answer word directly (not the question)
        from models.constants import TRANSLATIONS
        answer_word = TRANSLATIONS.get(cat, {}).get(lang, "?")
        full_ids  = processor.tokenizer.encode(f"{prefix} {answer_word}", add_special_tokens=False)
        word_ids  = full_ids[len(prefix_ids):]
        stripped  = [i for i in word_ids if i != 29871]
        used_id   = stripped[0] if stripped else None
        decoded_full    = [processor.tokenizer.decode([i]) for i in word_ids]
        decoded_used    = processor.tokenizer.decode([used_id]) if used_id else "NONE"
        multi_token_flag = "⚠ MULTI-TOKEN" if len(stripped) > 1 else ""
        print(f"[TOK] {cat:<14} {lang}  word='{answer_word}'  "
              f"all_tokens={decoded_full}  "
              f"used='{decoded_used}' (id={used_id})  {multi_token_flag}")
print("[TOK] ────────────────────────────────────────────────────────────────\n")

for cond_idx, use_image in enumerate([True, False]):
    for lang_idx, lang in enumerate(LANGUAGES):
        for ex_idx, ex in enumerate(tqdm(examples, desc=f"cond={cond_idx} lang={lang}")):
            en_tid = ex["token_ids"].get("en")
            tl_tid = ex["token_ids"].get(lang)
            if en_tid is None or tl_tid is None:
                continue
            img = Image.open(ex["image_path"]).convert("RGB")
            question = ex["questions"][lang]

            en_word  = processor.tokenizer.decode([en_tid])
            tl_word  = processor.tokenizer.decode([tl_tid])
            img_info = ex["image_path"] if use_image else "blank"
            print(f"[EXP1] cond={'image' if use_image else 'blank'} lang={lang} ex={ex_idx} "
                  f"| image={img_info} | q={question} "
                  f"| en_ans='{en_word}' tl_ans='{tl_word}'")

            if use_image:
                inputs, answer_pos = get_image_text_inputs(processor, img, question)
            else:
                inputs, answer_pos = get_blank_image_inputs(processor, question)

            residuals = extract_residual_streams(model, inputs, layers=list(range(NUM_LAYERS)))

            for layer in range(NUM_LAYERS):
                h = residuals[layer][answer_pos]             # (4096,)
                probs = apply_logit_lens(model, h)           # (32000,)
                log_probs = np.log(probs + 1e-10)
                # PMI = log P(token|layer h) - log P(token|zero h)
                en_probs[cond_idx, lang_idx, ex_idx, layer] = (
                    log_probs[en_tid] - baseline_log_probs[en_tid])
                tl_probs[cond_idx, lang_idx, ex_idx, layer] = (
                    log_probs[tl_tid] - baseline_log_probs[tl_tid])

np.save("outputs/logit_lens/logit_lens_en_prob.npy", en_probs)
np.save("outputs/logit_lens/logit_lens_tl_prob.npy", tl_probs)
np.save("outputs/logit_lens/logit_lens_mean_en.npy", np.nanmean(en_probs, axis=2))
np.save("outputs/logit_lens/logit_lens_mean_tl.npy", np.nanmean(tl_probs, axis=2))
json.dump(
    {"N": N, "languages": LANGUAGES, "conditions": ["image", "blank"]},
    open("outputs/logit_lens/exp1_metadata.json", "w")
)
print("DONE: Experiment 1")
