import os
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import (load_model, get_image_text_inputs,
                                   extract_residual_streams, extract_submodule_streams,
                                   apply_logit_lens, patch_and_run, patch_submodule_and_run)
from models.constants import LANGUAGES, NUM_LAYERS, VISUAL_START, VISUAL_END

os.makedirs("outputs/patching", exist_ok=True)
model, processor = load_model()
examples = json.load(open("data/probing_set/probing_set.json"))

# ── Example I/O diagnostic ────────────────────────────────────────────────────
print("\n[IO] ── Example input/output diagnostic ──")
_ex0 = examples[0]
_img0 = Image.open(_ex0["image_path"]).convert("RGB")
print(f"[IO] Image path : {_ex0['image_path']}")
print(f"[IO] Image size : {_img0.size}  (W×H)")
print(f"[IO] Category   : {_ex0.get('category', 'n/a')}")
for _lg in ["en", "fr", "ar", "zh", "bn"]:
    if _lg in _ex0.get("questions", {}):
        print(f"[IO] Question [{_lg}]: {_ex0['questions'][_lg]}")
for _lg in ["en", "fr", "ar", "zh", "bn"]:
    if _lg in _ex0.get("token_ids", {}):
        _tid = _ex0["token_ids"][_lg]
        _word = processor.tokenizer.decode([_tid])
        print(f"[IO] Target token [{_lg}]: id={_tid}  decoded='{_word}'")
# Run greedy generation on the English question to see what the model outputs
_inp_gen, _ = get_image_text_inputs(processor, _img0, _ex0["questions"]["en"])
with torch.no_grad():
    _gen_ids = model.generate(**_inp_gen, max_new_tokens=10, do_sample=False)
_gen_text = processor.decode(_gen_ids[0][_inp_gen["input_ids"].shape[1]:],
                              skip_special_tokens=True)
print(f"[IO] Model output (en question, greedy): '{_gen_text}'")
del _ex0, _img0, _inp_gen, _gen_ids, _gen_text
print("[IO] ────────────────────────────────────────\n")

# ── Patching sanity check ─────────────────────────────────────────────────────
print("[SANITY] Verifying activation patching works...")
_ex = examples[0]
_img = Image.open(_ex["image_path"]).convert("RGB")
_inputs_clean, _ans = get_image_text_inputs(processor, _img, _ex["questions"]["fr"])
_inputs_corr,  _    = get_image_text_inputs(processor, _img, _ex["questions"]["en"])
_lc, _lr = _inputs_clean["input_ids"].shape[1], _inputs_corr["input_ids"].shape[1]
_ml = min(_lc, _lr)
print(f"[SANITY] seq_len clean(fr)={_lc}  corrupted(en)={_lr}  answer_pos={_ans}  min_len={_ml}")
# Only truncate corrupted — keep clean at full length so answer_pos stays at ASSISTANT:
for _k in ("input_ids", "attention_mask"):
    _inputs_corr[_k] = _inputs_corr[_k][:, :_ml]
print(f"[SANITY] answer_pos kept at {_ans} (last token of ASSISTANT: in clean run)")

# Patch only positions that exist in both sequences
_all_pos = list(range(_ml))
_en_tid  = _ex["token_ids"]["en"]
_fr_tid  = _ex["token_ids"].get("fr")
print(f"[SANITY] en_tid={_en_tid}  fr_tid={_fr_tid}")

# Step 1: check what the clean (French) run looks like at last layer
_clean_cache = extract_residual_streams(model, _inputs_clean, [31])
_clean_h31   = _clean_cache[31][_ans]
_probs_clean = apply_logit_lens(model, _clean_h31)
_top5_clean  = sorted(enumerate(_probs_clean), key=lambda x: -x[1])[:5]
print(f"[SANITY] Clean run top-5 tokens at answer_pos:")
for _tid, _tp in _top5_clean:
    print(f"  id={_tid:6d}  p={_tp:.6f}  word='{processor.tokenizer.decode([_tid])}'")
print(f"[SANITY] P(en='cat') in clean run = {float(_probs_clean[_en_tid]):.6f}")

# Step 2: top-5 of corrupted (English) run and find its first response token
_corr_cache = extract_residual_streams(model, _inputs_corr, [16, 31])
_corr_h16   = _corr_cache[16]
_corr_h31   = _corr_cache[31][-1]
_probs_corr = apply_logit_lens(model, _corr_h31)
_top5_corr  = sorted(enumerate(_probs_corr), key=lambda x: -x[1])[:5]
print(f"[SANITY] Corrupted (en) run top-5 tokens at answer_pos:")
for _tid, _tp in _top5_corr:
    print(f"  id={_tid:6d}  p={_tp:.6f}  word='{processor.tokenizer.decode([_tid])}'")

# The first generated token is what the model ACTUALLY outputs — use this as target,
# not the object label ('cat') which is never the first generated token.
with torch.no_grad():
    _en_resp_ids = model.generate(**_inputs_corr, max_new_tokens=1, do_sample=False)
_en_resp_tid = int(_en_resp_ids[0, -1])
with torch.no_grad():
    _fr_resp_ids = model.generate(**_inputs_clean, max_new_tokens=1, do_sample=False)
_fr_resp_tid = int(_fr_resp_ids[0, -1])
print(f"[SANITY] English first response token: id={_en_resp_tid}  word='{processor.tokenizer.decode([_en_resp_tid])}'")
print(f"[SANITY] French  first response token: id={_fr_resp_tid}  word='{processor.tokenizer.decode([_fr_resp_tid])}'")
print(f"[SANITY] P(en_first_token) in clean run   = {float(_probs_clean[_en_resp_tid]):.6f}")
print(f"[SANITY] P(en_first_token) in corrupt run = {float(_probs_corr[_en_resp_tid]):.6f}")

# Step 3: verify hidden states differ
_clean_h16 = extract_residual_streams(model, _inputs_clean, [16])[16]
_diff = np.abs(_clean_h16[:_ml] - _corr_h16[:_ml]).mean()
print(f"[SANITY] Mean abs diff clean/corrupted at layer 16: {_diff:.6f}  "
      f"{'(ZERO — identical inputs?!)' if _diff < 1e-6 else '(non-zero, good)'}")

# Step 4: pre-hook fires?
_hook_fired = {"fired": False}
def _test_pre_hook(module, args):
    _hook_fired["fired"] = True
    return args
_test_h = model.language_model.model.layers[17].register_forward_pre_hook(_test_pre_hook)
with torch.no_grad():
    model(**_inputs_clean)
_test_h.remove()
print(f"[SANITY] Pre-hook on layer 17 fired: {_hook_fired['fired']}")

# Step 5: patch test using en_resp_tid (first English response token, not object label)
_p_clean   = float(_probs_clean[_en_resp_tid])
_p_patched = float(patch_and_run(model, _inputs_clean, _inputs_corr, 16,
                                  _all_pos, corrupted_cache=_corr_cache)[_en_resp_tid])
_delta = _p_patched - _p_clean
print(f"[SANITY] tracking token '{processor.tokenizer.decode([_en_resp_tid])}'")
print(f"[SANITY] p_en_resp clean={_p_clean:.6f}  patched={_p_patched:.6f}  delta={_delta:+.6f}")
if abs(_delta) < 1e-4:
    print("[SANITY] WARNING: delta ~0 — patching still not working.")
else:
    print("[SANITY] OK — patching produces non-zero effect, proceeding.")

del _ex, _img, _inputs_clean, _inputs_corr, _corr_cache, _clean_cache, _clean_h31
del _clean_h16, _corr_h16, _corr_h31, _all_pos, _en_tid, _fr_tid
del _lc, _lr, _ans, _p_clean, _p_patched, _delta, _diff, _hook_fired
del _en_resp_tid, _fr_resp_tid, _en_resp_ids, _fr_resp_ids, _ml
# ─────────────────────────────────────────────────────────────────────────────

# WARNING: ~38k forward passes total. Expect 12-20h on A100.
# To do a quick test run, replace examples with examples[:5].

aie_residual = np.zeros((4, NUM_LAYERS), dtype=np.float32)
aie_attn     = np.zeros((4, NUM_LAYERS), dtype=np.float32)
aie_mlp      = np.zeros((4, NUM_LAYERS), dtype=np.float32)
aie_visual   = np.zeros((4, NUM_LAYERS), dtype=np.float32)
counts       = np.zeros(4, dtype=np.int32)

for lang_idx, lang in enumerate(LANGUAGES):
    for ex in tqdm(examples, desc=f"Patching [{lang}]"):
        img = Image.open(ex["image_path"]).convert("RGB")

        inputs_clean, answer_pos = get_image_text_inputs(processor, img, ex["questions"][lang])
        inputs_corr,  _          = get_image_text_inputs(processor, img, ex["questions"]["en"])

        # Only truncate corrupted to min_len — keep clean at full length so
        # answer_pos stays at the real last token (end of ASSISTANT: prompt).
        len_clean = inputs_clean["input_ids"].shape[1]
        len_corr  = inputs_corr["input_ids"].shape[1]
        min_len   = min(len_clean, len_corr)
        inputs_corr["input_ids"]       = inputs_corr["input_ids"][:, :min_len]
        inputs_corr["attention_mask"]  = inputs_corr["attention_mask"][:, :min_len]

        # Use the first generated token of the English run as the target.
        # P('cat') ≈ 0 at answer_pos because the model generates "The image shows a cat"
        # — 'cat' is never the first token. The first token (e.g. 'The') is what
        # answer_pos actually predicts and is a reliable language-identity signal.
        with torch.no_grad():
            _resp = model.generate(**inputs_corr, max_new_tokens=1, do_sample=False)
        en_tid = int(_resp[0, -1])

        # Baseline: P(en first response token) in clean (non-English) run
        res_clean = extract_residual_streams(model, inputs_clean, [NUM_LAYERS - 1])
        probs_clean = apply_logit_lens(model, res_clean[NUM_LAYERS - 1][answer_pos])
        p_en_clean = float(probs_clean[en_tid])

        # Patch only positions that exist in both sequences
        all_positions    = list(range(min_len))
        visual_positions = list(range(VISUAL_START, min(VISUAL_END + 1, min_len)))

        # Pre-extract ALL corrupted layers once (residual, attn, mlp) — 3 passes total
        # instead of 1 + 32 + 32 = 65 corrupted passes per example.
        corrupted_all  = extract_residual_streams(model, inputs_corr, list(range(NUM_LAYERS)))
        corrupted_attn = extract_submodule_streams(model, inputs_corr, list(range(NUM_LAYERS)), "attn")
        corrupted_mlp  = extract_submodule_streams(model, inputs_corr, list(range(NUM_LAYERS)), "mlp")

        torch.cuda.empty_cache()

        for L in range(NUM_LAYERS):
            # Full residual patch
            probs = patch_and_run(model, inputs_clean, inputs_corr, L, all_positions,
                                  corrupted_cache=corrupted_all)
            aie_residual[lang_idx, L] += float(probs[en_tid]) - p_en_clean

            # Visual-only patch
            probs_vis = patch_and_run(model, inputs_clean, inputs_corr, L, visual_positions,
                                      corrupted_cache=corrupted_all)
            aie_visual[lang_idx, L] += float(probs_vis[en_tid]) - p_en_clean

            # Attention-output patch
            probs_attn = patch_submodule_and_run(
                model, inputs_clean, inputs_corr, L, "attn", all_positions,
                corrupted_cache=corrupted_attn
            )
            aie_attn[lang_idx, L] += float(probs_attn[en_tid]) - p_en_clean

            # MLP-output patch
            probs_mlp = patch_submodule_and_run(
                model, inputs_clean, inputs_corr, L, "mlp", all_positions,
                corrupted_cache=corrupted_mlp
            )
            aie_mlp[lang_idx, L] += float(probs_mlp[en_tid]) - p_en_clean

        counts[lang_idx] += 1

# Normalise by example count
for i in range(4):
    if counts[i] > 0:
        aie_residual[i] /= counts[i]
        aie_attn[i]     /= counts[i]
        aie_mlp[i]      /= counts[i]
        aie_visual[i]   /= counts[i]

np.save("outputs/patching/patching_residual.npy", aie_residual)
np.save("outputs/patching/patching_attn.npy",     aie_attn)
np.save("outputs/patching/patching_mlp.npy",      aie_mlp)
np.save("outputs/patching/patching_visual.npy",   aie_visual)

# Identify pivot layers from mean residual AIE across languages
mean_aie = aie_residual.mean(axis=0)                   # shape (32,)
print("[DEBUG] mean AIE per layer:")
for i, v in enumerate(mean_aie):
    print(f"  layer {i:2d}: {v:+.6f}")
print(f"[DEBUG] AIE range: min={mean_aie.min():.6f}  max={mean_aie.max():.6f}")

peak = float(mean_aie.max())
if peak > 0:
    # Normal case: layers with AIE >= 50% of peak
    threshold = 0.5 * peak
    pivot_layers = sorted([int(i) for i in np.where(mean_aie >= threshold)[0]])
else:
    # All AIE non-positive → patching showed no benefit at any layer.
    # Fall back to top-3 layers by AIE so downstream experiments can still run.
    print("[WARNING] All mean AIE values are non-positive — no clear translation "
          "layers found. Using top-3 layers as pivot_layers fallback.")
    pivot_layers = sorted(int(i) for i in np.argsort(mean_aie)[-3:])

json.dump(
    {"pivot_layers": pivot_layers, "mean_aie_per_layer": mean_aie.tolist()},
    open("outputs/pivot_layers.json", "w"), indent=2
)
print(f"Pivot layers: {pivot_layers}")
print("DONE: Experiment 2")
