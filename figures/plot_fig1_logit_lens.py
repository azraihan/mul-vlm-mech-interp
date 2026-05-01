import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANGUAGES, LANG_NAMES

N_LANGS = len(LANGUAGES)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
os.makedirs("outputs/figures", exist_ok=True)

mean_en = np.load("outputs/logit_lens/logit_lens_mean_en.npy")   # (2, 4, 32)
mean_tl = np.load("outputs/logit_lens/logit_lens_mean_tl.npy")   # (2, 4, 32)
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]

fig, axes = plt.subplots(2, N_LANGS, figsize=(4 * N_LANGS, 8), sharey=True)
layers = list(range(32))

handles = []
for cond in range(2):
    for lang in range(N_LANGS):
        ax = axes[cond, lang]
        en_curve = mean_en[cond, lang, :]
        tl_curve = mean_tl[cond, lang, :]

        l1, = ax.plot(layers, en_curve, color="#2166ac", linewidth=2, label="English")
        l2, = ax.plot(layers, tl_curve, color="#d6604d", linewidth=2, linestyle="--",
                      label=LANG_NAMES[LANGUAGES[lang]])
        ax.axvspan(min(pivot_layers) - 0.5, max(pivot_layers) + 0.5,
                   alpha=0.12, color="gray",
                   label="Pivot region" if (cond == 0 and lang == 0) else "")
        ax.set_xlim(0, 31)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Transformer Layer")
        if lang == 0:
            ax.set_ylabel("Log PMI")
        title_cond = "Image Present" if cond == 0 else "Blank Image"
        ax.set_title(f"{LANG_NAMES[LANGUAGES[lang]]}\n({title_cond})", fontsize=10)

        if cond == 0 and lang == 0:
            handles = [l1, l2, mpatches.Patch(color="gray", alpha=0.3, label="Pivot region")]

# Second pass: add overlays after all curves are drawn so sharey ylim is finalized
for cond in range(2):
    for lang in range(N_LANGS):
        ax = axes[cond, lang]
        en_curve = mean_en[cond, lang, :]
        tl_curve = mean_tl[cond, lang, :]
        ylim = ax.get_ylim()

        # vertical line from x-axis spine (ymin=0 fraction) to each curve's peak
        for curve, color in [(en_curve, "#2166ac"), (tl_curve, "#d6604d")]:
            peak_layer = int(np.argmax(curve))
            peak_val = curve[peak_layer]
            ymax_frac = (peak_val - ylim[0]) / (ylim[1] - ylim[0])
            ax.axvline(peak_layer, ymin=0, ymax=ymax_frac, color=color,
                       linewidth=1.5, linestyle=":", alpha=0.75)

        # endpoint dot + value label at layer 31 — both above their dot
        last_layer = 31
        for curve, color in [(en_curve, "#2166ac"), (tl_curve, "#d6604d")]:
            y_end = curve[last_layer]
            ax.plot(last_layer, y_end, "o", color=color, markersize=5,
                    zorder=5, clip_on=False)
            ax.annotate(f"{y_end:.2f}", xy=(last_layer, y_end),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=color,
                        annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.85))

fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()

plt.savefig("outputs/figures/fig1_logit_lens.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig1_logit_lens.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig1_logit_lens.pdf/png")
