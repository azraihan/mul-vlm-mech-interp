import os
import sys
import json
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import (load_model, get_image_text_inputs,
                                   extract_residual_streams, apply_logit_lens,
                                   compute_pmi_baseline)
from models.constants import LANGUAGES, LANG_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
os.makedirs("outputs/figures", exist_ok=True)

# Load data
examples = json.load(open("data/probing_set/probing_set.json"))
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]
en_probs_all = np.load("outputs/logit_lens/logit_lens_en_prob.npy")   # (2, 4, N, 32)
tl_probs_all = np.load("outputs/logit_lens/logit_lens_tl_prob.npy")   # (2, 4, N, 32)

model, processor = load_model()
baseline_log_probs = compute_pmi_baseline(model)   # (32000,)

# STEP 1: Pick examples — for ar and bn, find the most severe English pivot
chosen = {}
for lang_label, lang in [("ar", "ar"), ("bn", "bn")]:
    lang_idx = LANGUAGES.index(lang)
    # max P(en) - max P(tl) across layers, for cond=0 (image present)
    en_max = en_probs_all[0, lang_idx, :, :].max(axis=1)   # (N,)
    tl_max = tl_probs_all[0, lang_idx, :, :].max(axis=1)   # (N,)
    diff = en_max - tl_max
    best_idx = int(np.argmax(diff))
    chosen[lang] = (best_idx, examples[best_idx])

# STEP 2 & 3: Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
col_keys = ["ar", "bn"]

# Accumulate curve data for serialization — lets you restyle without re-running the model.
fig5_data = {}

for col_i, lang in enumerate(col_keys):
    lang_idx = LANGUAGES.index(lang)
    ex_idx, ex = chosen[lang]
    img = Image.open(ex["image_path"]).convert("RGB")
    question = ex["questions"][lang]
    en_tid = ex["token_ids"]["en"]
    tl_tid = ex["token_ids"][lang]

    inputs, answer_pos = get_image_text_inputs(processor, img, question)

    sv_path = f"outputs/steering_vectors/steering_vec_{lang}.pt"
    sv = torch.load(sv_path, map_location="cuda") if os.path.exists(sv_path) else None

    fig5_data[lang] = {
        "lang": lang,
        "lang_name": LANG_NAMES[lang],
        "example_idx": ex_idx,
        "image_path": ex["image_path"],
        "category": ex["category"],
        "question": question,
        "en_token_id": en_tid,
        "tl_token_id": tl_tid,
        "curves": {}
    }

    for row_i, use_steering in enumerate([False, True]):
        ax = axes[row_i, col_i]
        condition = "steered" if use_steering else "unsteered"

        if use_steering and sv is not None:
            def make_steer_hook(L, vec, ap):
                def hook_fn(module, inp, output):
                    h = output[0].clone()
                    v = vec[L].to(h.device).to(h.dtype)
                    h[:, ap:, :] += 1.0 * v
                    return (h,) + output[1:]
                return hook_fn

            hooks = []
            for L in pivot_layers:
                if L in sv:
                    hooks.append(
                        model.language_model.model.layers[L].register_forward_hook(
                            make_steer_hook(L, sv, answer_pos)
                        )
                    )
            try:
                residuals = extract_residual_streams(model, inputs, list(range(32)))
            finally:
                for h in hooks:
                    h.remove()
        else:
            residuals = extract_residual_streams(model, inputs, list(range(32)))

        en_curve = np.array([
            np.log(apply_logit_lens(model, residuals[l][answer_pos])[en_tid] + 1e-10)
            - baseline_log_probs[en_tid] for l in range(32)
        ])
        tl_curve = np.array([
            np.log(apply_logit_lens(model, residuals[l][answer_pos])[tl_tid] + 1e-10)
            - baseline_log_probs[tl_tid] for l in range(32)
        ])

        # Persist curves so this figure can be restyled without re-running the model.
        fig5_data[lang]["curves"][condition] = {
            "en_prob_per_layer": en_curve.tolist(),
            "tl_prob_per_layer": tl_curve.tolist(),
        }

        layers = list(range(32))
        ax.plot(layers, en_curve, color="#2166ac", linewidth=2, label="English")
        ax.plot(layers, tl_curve, color="#d6604d", linewidth=2, linestyle="--",
                label=LANG_NAMES[lang])
        ax.set_xlim(0, 31)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Log PMI")
        steering_str = "With steering" if use_steering else "Without steering"
        ax.set_title(f"{LANG_NAMES[lang]} — {steering_str}", fontsize=10)

        inset = ax.inset_axes([0.02, 0.62, 0.18, 0.35])
        inset.imshow(np.array(img))
        inset.axis("off")

        crossover = next((i for i in range(32) if tl_curve[i] > en_curve[i]), None)
        if crossover is not None:
            ax.annotate("↑", xy=(crossover, tl_curve[crossover]), fontsize=14, color="red",
                        ha="center")

# Save the underlying curve data separately so the figure can be restyled without the model.
with open("outputs/figures/fig5_data.json", "w", encoding="utf-8") as f:
    json.dump(fig5_data, f, ensure_ascii=False, indent=2)
print("Saved outputs/figures/fig5_data.json")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()

plt.savefig("outputs/figures/fig5_qualitative.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig5_qualitative.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig5_qualitative.pdf/png")
