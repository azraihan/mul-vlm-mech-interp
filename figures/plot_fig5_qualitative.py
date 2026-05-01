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

OUT_PDF = "outputs/figures/fig5_qualitative.pdf"
OUT_PNG = "outputs/figures/fig5_qualitative.png"
OUT_DATA = "outputs/figures/fig5_qualitative_data.json"
os.makedirs("outputs/figures", exist_ok=True)

# ── Selected examples: (lang, probing_set_index, expected_category) ──────────
SELECTED = [
    ("ru", 61,  "boat"),
    ("de", 75,  "orange"),
    ("pt", 126, "spoon"),
]

examples     = json.load(open("data/probing_set/probing_set.json"))
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]

# Validate selections
for lang, idx, expected_cat in SELECTED:
    ex = examples[idx]
    assert ex["category"] == expected_cat, (
        f"Mismatch at index {idx}: expected '{expected_cat}', got '{ex['category']}'"
    )
print("Selections validated:", [(lang, idx, cat) for lang, idx, cat in SELECTED])

model, processor = load_model()
baseline_log_probs = compute_pmi_baseline(model)

# Load steering vectors
steering_vecs = {}
for lang in LANGUAGES:
    sv_path = f"outputs/steering_vectors/steering_vec_{lang}.pt"
    if os.path.exists(sv_path):
        steering_vecs[lang] = torch.load(sv_path, map_location="cuda")
        print(f"[SV] Loaded steering vector for {lang}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=None):
    """Forward pass (optionally steered) → (en_curve, tl_curve), each shape (32,)."""
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
        - baseline_log_probs[en_tid] for l in range(32)
    ])
    tl_c = np.array([
        np.log(apply_logit_lens(model, residuals[l][answer_pos])[tl_tid] + 1e-10)
        - baseline_log_probs[tl_tid] for l in range(32)
    ])
    return en_c, tl_c


def plot_curve(ax, en_curve, tl_curve, lang, title):
    layers = list(range(32))

    ax.axvspan(min(pivot_layers) - 0.5, max(pivot_layers) + 0.5,
               alpha=0.08, color="gray")
    ax.plot(layers, en_curve, color="#2166ac", linewidth=1.8, label="English")
    ax.plot(layers, tl_curve, color="#d6604d", linewidth=1.8, linestyle="--",
            label=LANG_NAMES[lang])
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlim(0, 31)
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("Log PMI", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7.5, loc="upper left", handlelength=1.5, framealpha=0.7)

    crossover = next((i for i in range(32) if tl_curve[i] > en_curve[i]), None)
    if crossover is not None:
        y_cross = tl_curve[crossover]

        # Capture current ylim so the drop-line doesn't alter autoscaling
        y_lo, y_hi = ax.get_ylim()

        # Vertical dashed drop-line from bottom axis to crossover point
        ax.plot([crossover, crossover], [y_lo, y_cross],
                color="black", linestyle="--", linewidth=1.0, zorder=3,
                clip_on=True)

        # Restore ylim in case the new line nudged it
        ax.set_ylim(y_lo, y_hi)

        # Layer number below the bottom x-axis (outside the plot area)
        ax.annotate(
            f"L{crossover}",
            xy=(crossover, 0),                  # anchor: x in data, y=0 = bottom of axes
            xycoords=("data", "axes fraction"),
            xytext=(0, -12),                    # shift 12 pts downward
            textcoords="offset points",
            ha="center", va="top", fontsize=7.5,
            color="black", fontweight="bold", zorder=4,
            annotation_clip=False,
        )

        # Longer red upward arrow above the crossover point
        curve_range = np.nanmax(np.concatenate([en_curve, tl_curve])) \
                    - np.nanmin(np.concatenate([en_curve, tl_curve]))
        arrow_len = max(curve_range * 0.18, 0.3)
        ax.annotate(
            "",
            xy=(crossover, y_cross + arrow_len),   # arrow head
            xytext=(crossover, y_cross),            # arrow tail
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                lw=2.0,
                mutation_scale=14,
            ),
            zorder=5,
        )

    return crossover


# ── Compute all curves ────────────────────────────────────────────────────────

all_data = {}
rows = []   # (lang, img, en_word, tl_word, en_u, tl_u, en_s, tl_s)

for lang, idx, cat in SELECTED:
    ex     = examples[idx]
    en_tid = ex["token_ids"]["en"]
    tl_tid = ex["token_ids"].get(lang)
    img    = Image.open(ex["image_path"]).convert("RGB")
    q      = ex["questions"][lang]

    inputs, answer_pos = get_image_text_inputs(processor, img, q)
    sv = steering_vecs.get(lang)

    print(f"\n[{lang}] idx={idx} cat={cat}")
    en_u, tl_u = compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=None)
    en_s, tl_s = compute_curves(inputs, answer_pos, en_tid, tl_tid, sv=sv)
    print(f"  done")

    en_word = TRANSLATIONS[cat]["en"]
    tl_word = TRANSLATIONS[cat][lang]
    rows.append((lang, img, en_word, tl_word, en_u, tl_u, en_s, tl_s))

    all_data[lang] = {
        "lang":          lang,
        "lang_name":     LANG_NAMES[lang],
        "example_index": idx,
        "category":      cat,
        "image_path":    ex["image_path"],
        "question":      q,
        "en_token_id":   en_tid,
        "tl_token_id":   tl_tid,
        "en_word":       en_word,
        "tl_word":       tl_word,
        "pivot_layers":  pivot_layers,
        "curves": {
            "unsteered": {"en": en_u.tolist(), "tl": tl_u.tolist()},
            "steered":   {"en": en_s.tolist(), "tl": tl_s.tolist()},
        },
    }

with open(OUT_DATA, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
print(f"\nSaved {OUT_DATA}")

# ── Figure ────────────────────────────────────────────────────────────────────
# 3 rows (ru / de / pt).  Each row: [image+caption | baseline | steered]
# Left col uses a nested GridSpec to split image (tall) from caption (short).

n_rows = len(rows)
fig = plt.figure(figsize=(17, 4.2 * n_rows))
outer = gridspec.GridSpec(
    n_rows, 3,
    figure=fig,
    width_ratios=[1, 2.2, 2.2],
    hspace=0.55,
    wspace=0.32,
)

for row_i, (lang, img, en_word, tl_word, en_u, tl_u, en_s, tl_s) in enumerate(rows):

    # Left sub-grid: image (tall) + caption+legend area (shorter)
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row_i, 0],
        height_ratios=[4, 2], hspace=0.06,
    )

    ax_img = fig.add_subplot(left_gs[0])
    ax_img.imshow(np.array(img))
    ax_img.axis("off")

    ax_cap = fig.add_subplot(left_gs[1])
    ax_cap.axis("off")

    # Caption text at top of the area
    ax_cap.text(
        0.5, 0.98,
        f"{en_word} ({tl_word})",
        transform=ax_cap.transAxes,
        fontsize=9, va="top", ha="center",
        style="italic",
    )

    # Baseline graph
    ax_base = fig.add_subplot(outer[row_i, 1])
    plot_curve(ax_base, en_u, tl_u, lang, f"Baseline  ·  {LANG_NAMES[lang]}")

    # Steered graph
    ax_steer = fig.add_subplot(outer[row_i, 2])
    plot_curve(ax_steer, en_s, tl_s, lang, f"Steered  ·  {LANG_NAMES[lang]}")

    # Row legend in the caption area, below the caption text
    handles, labels = ax_base.get_legend_handles_labels()
    for ax in [ax_base, ax_steer]:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    ax_cap.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=8.5,
        handlelength=1.8,
        framealpha=0,
        borderpad=0,
    )

plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")