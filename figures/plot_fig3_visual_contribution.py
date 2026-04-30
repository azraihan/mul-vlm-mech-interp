import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANGUAGES, LANG_NAMES

N_LANGS = len(LANGUAGES)

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

residual = np.load("outputs/patching/patching_residual.npy")   # (4, 32)
visual   = np.load("outputs/patching/patching_visual.npy")     # (4, 32)

fig, axes = plt.subplots(1, N_LANGS, figsize=(4 * N_LANGS, 4), sharey=True)
layers = list(range(32))

for lang_idx in range(N_LANGS):
    ax = axes[lang_idx]
    ax.plot(layers, residual[lang_idx], color="#2166ac", linewidth=2, label="Full patch")
    ax.plot(layers, visual[lang_idx],   color="#4dac26", linewidth=2, linestyle="--",
            label="Visual tokens only")
    ax.fill_between(layers, visual[lang_idx], residual[lang_idx],
                    alpha=0.15, color="#7b3294", label="Text token contribution")
    ax.set_xlabel("Layer")
    if lang_idx == 0:
        ax.set_ylabel("AIE")
    ax.set_title(LANG_NAMES[LANGUAGES[lang_idx]])
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()

# Inset on Russian subplot: zoom into visual-only curve
from matplotlib.patches import ConnectionPatch
ru_idx = LANGUAGES.index("ru")
ax_ru = axes[ru_idx]
vis_ru = visual[ru_idx]

inset_pos = [0.63, 0.62, 0.33, 0.25]  # top-right
inset = ax_ru.inset_axes(inset_pos)
inset.plot(layers, vis_ru, color="#4dac26", linewidth=1.2)
inset.axhline(0, color="black", linewidth=0.5, linestyle=":")
inset.set_xlim(0, 31)
inset.set_ylim(-0.0002, 0.0002)
inset.set_yticks([-0.0001, 0, 0.0001])
inset.set_yticklabels(["-1e-4", "0", "1e-4"], fontsize=7)
inset.set_xticks([0, 16, 31])
inset.set_xticklabels(["0", "16", "31"], fontsize=7)
inset.set_title("Visual only (zoomed)", fontsize=7, pad=3)

# Both connector lines converge to the peak of the visual signal in the main plot
focus_layer = int(np.argmax(np.abs(vis_ru)))
ix0, iy0, iw, ih = inset_pos
for corner_x in [ix0, ix0 + iw]:
    con = ConnectionPatch(
        xyA=(corner_x, iy0), coordsA="axes fraction",
        xyB=(focus_layer, 0.0), coordsB="data",
        axesA=ax_ru, axesB=ax_ru,
        color="gray", alpha=0.5, linewidth=0.8,
    )
    ax_ru.add_patch(con)

plt.savefig("outputs/figures/fig3_visual_contribution.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig3_visual_contribution.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig3_visual_contribution.pdf/png")
