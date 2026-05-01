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
from models.constants import LANGUAGES, LANG_NAMES, TRANSLATIONS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT_DIR = "outputs/figures/fig5"
os.makedirs(OUT_DIR, exist_ok=True)

examples = json.load(open("data/probing_set/probing_set.json"))
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]

# Determine which examples still need generating
pending = [
    (ex_idx, ex) for ex_idx, ex in enumerate(examples)
    if not os.path.exists(f"{OUT_DIR}/{ex_idx:03d}_{ex['category']}.png")
]
print(f"{len(pending)} / {len(examples)} figures to generate.")
if not pending:
    print("All done.")
    sys.exit(0)

model, processor = load_model()
baseline_log_probs = compute_pmi_baseline(model)

# Load all steering vectors once upfront
steering_vecs = {}
for lang in LANGUAGES:
    sv_path = f"outputs/steering_vectors/steering_vec_{lang}.pt"
    if os.path.exists(sv_path):
        steering_vecs[lang] = torch.load(sv_path, map_location="cuda")
        print(f"[SV] Loaded steering vector for {lang}")


def compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=None):
    """
    Run one forward pass (optionally steered at pivot layers) and return
    (en_curve, tl_curve) as PMI log-prob arrays of shape (32,).
    """
    hooks = []
    if sv is not None:
        def make_hook(L, vec):
            def fn(module, inp, output):
                t = output[0] if isinstance(output, tuple) else output
                v = vec[L].to(t.device).to(t.dtype)
                if t.dim() == 3:
                    t.data[:, -1:, :] += v
                else:
                    t.data[-1:, :] += v
            return fn
        for L in pivot_layers:
            if L in sv:
                hooks.append(
                    model.language_model.model.layers[L].register_forward_hook(
                        make_hook(L, sv)
                    )
                )
    try:
        residuals = extract_residual_streams(model, inputs, list(range(32)))
    finally:
        for h in hooks:
            h.remove()

    en_c = np.array([
        np.log(apply_logit_lens(model, residuals[l][answer_pos])[en_tid] + 1e-10)
        - baseline_log_probs[en_tid]
        for l in range(32)
    ])
    tl_c = np.array([
        np.log(apply_logit_lens(model, residuals[l][answer_pos])[tl_tid] + 1e-10)
        - baseline_log_probs[tl_tid]
        for l in range(32)
    ])
    return en_c, tl_c


def plot_curve(ax, en_curve, tl_curve, lang, title):
    layers = list(range(32))
    ax.plot(layers, en_curve, color="#2166ac", linewidth=1.8, label="English")
    ax.plot(layers, tl_curve, color="#d6604d", linewidth=1.8, linestyle="--",
            label=LANG_NAMES[lang])
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlim(0, 31)
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("Log PMI", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    crossover = next((i for i in range(32) if tl_curve[i] > en_curve[i]), None)
    if crossover is not None:
        ax.annotate("↑", xy=(crossover, tl_curve[crossover]),
                    fontsize=12, color="red", ha="center")


for ex_idx, ex in pending:
    out_path = f"{OUT_DIR}/{ex_idx:03d}_{ex['category']}.png"
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path}")
        continue

    img = Image.open(ex["image_path"]).convert("RGB")
    en_tid = ex["token_ids"]["en"]
    cat = ex["category"]
    print(f"\n[{ex_idx:03d}/{len(examples)-1}] category={cat}")

    # Compute unsteered + steered curves for each language (2 forward passes × 3 langs = 6 total)
    curves = {}
    for lang in LANGUAGES:
        tl_tid = ex["token_ids"].get(lang)
        if tl_tid is None:
            print(f"  [{lang}] no token_id — skipping")
            continue
        inputs, answer_pos = get_image_text_inputs(processor, img, ex["questions"][lang])
        sv = steering_vecs.get(lang)
        en_u, tl_u = compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=None)
        en_s, tl_s = compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=sv)
        curves[lang] = (en_u, tl_u, en_s, tl_s)
        print(f"  [{lang}] done")

    # ── Figure layout ─────────────────────────────────────────────────────
    # 4 rows × 3 cols: rows 0-2 = one lang per row, row 3 = caption text
    # col 0 = image (spans rows 0-3), cols 1-2 = unsteered / steered plots
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        width_ratios=[1, 2, 2],
        height_ratios=[3, 3, 3, 1],
        hspace=0.55,
        wspace=0.3,
    )

    # Left: image spanning rows 0-2
    ax_img = fig.add_subplot(gs[:3, 0])
    ax_img.imshow(np.array(img))
    ax_img.axis("off")
    ax_img.set_title(f"#{ex_idx:03d}  —  {cat}", fontsize=10, pad=5)

    # Left: caption (answer word per language) in row 3
    ax_txt = fig.add_subplot(gs[3, 0])
    ax_txt.axis("off")
    trans = TRANSLATIONS.get(cat, {})
    caption = "  |  ".join(
        f"[{lg.upper()}] {trans.get(lg, '?')}"
        for lg in ["en"] + LANGUAGES
    )
    ax_txt.text(0.5, 0.85, caption, transform=ax_txt.transAxes,
                fontsize=8.5, va="top", ha="center", family="monospace")

    # Right: 3 lang rows × 2 condition cols
    for row_i, lang in enumerate(LANGUAGES):
        if lang not in curves:
            continue
        en_u, tl_u, en_s, tl_s = curves[lang]
        plot_curve(fig.add_subplot(gs[row_i, 1]),
                   en_u, tl_u, lang, f"{LANG_NAMES[lang]} — without steering")
        plot_curve(fig.add_subplot(gs[row_i, 2]),
                   en_s, tl_s, lang, f"{LANG_NAMES[lang]} — with steering")

    # Shared legend from first plot axes
    plot_axes = [ax for ax in fig.axes if ax not in (ax_img, ax_txt)]
    if plot_axes:
        handles, labels = plot_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.65, 0.0), fontsize=9)

    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved {out_path}")

print(f"\nDONE: figures written to {OUT_DIR}/")
