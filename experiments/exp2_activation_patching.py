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
        en_tid = ex["token_ids"]["en"]

        inputs_clean, answer_pos = get_image_text_inputs(processor, img, ex["questions"][lang])
        inputs_corr,  _          = get_image_text_inputs(processor, img, ex["questions"]["en"])

        # Ensure same seq_len (truncate to min length)
        len_clean = inputs_clean["input_ids"].shape[1]
        len_corr  = inputs_corr["input_ids"].shape[1]
        if len_corr != len_clean:
            min_len = min(len_clean, len_corr)
            inputs_clean["input_ids"]      = inputs_clean["input_ids"][:, :min_len]
            inputs_clean["attention_mask"] = inputs_clean["attention_mask"][:, :min_len]
            inputs_corr["input_ids"]       = inputs_corr["input_ids"][:, :min_len]
            inputs_corr["attention_mask"]  = inputs_corr["attention_mask"][:, :min_len]
            answer_pos = min_len - 1

        # Baseline: P(en token) in clean (non-English) run
        res_clean = extract_residual_streams(model, inputs_clean, [NUM_LAYERS - 1])
        probs_clean = apply_logit_lens(model, res_clean[NUM_LAYERS - 1][answer_pos])
        p_en_clean = float(probs_clean[en_tid])

        all_positions    = list(range(inputs_clean["input_ids"].shape[1]))
        visual_positions = list(range(VISUAL_START, VISUAL_END + 1))

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
threshold = 0.5 * mean_aie.max()
pivot_layers = sorted([int(i) for i in np.where(mean_aie >= threshold)[0]])
json.dump(
    {"pivot_layers": pivot_layers, "mean_aie_per_layer": mean_aie.tolist()},
    open("outputs/pivot_layers.json", "w"), indent=2
)
print(f"Pivot layers: {pivot_layers}")
print("DONE: Experiment 2")
