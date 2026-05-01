import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANGUAGES, LANG_NAMES

N_LANGS = len(LANGUAGES)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
os.makedirs("../outputs/figures", exist_ok=True)

patching_residual = np.load("../outputs/patching/patching_residual.npy")   # (4, 32)
patching_attn     = np.load("../outputs/patching/patching_attn.npy")       # (4, 32)
patching_mlp      = np.load("../outputs/patching/patching_mlp.npy")        # (4, 32)

arrays = [patching_residual, patching_attn, patching_mlp]
vmax = float(np.ceil(patching_residual.max() * 10) / 10)  # round up to nearest 0.1
col_titles = ["Residual Stream", "Attention Output", "MLP Output"]

# ── Overlay boxes ──────────────────────────────────────────────────────────
COLOR_CHOSEN = "#F2FF00"   # near-black      — chosen pivots (same all rows)
COLOR_LANG   = "#007525"   # dark green      — language-specific pivots

BOX_CHOSEN = (5, 17)       # x-range shared across all rows

# language-specific x-ranges indexed by lang_idx (row order matches LANGUAGES)
BOX_LANG = {0: (5.2, 14), 1: (5.2, 16.8), 2: (3, 16.8)}

fig, axes = plt.subplots(N_LANGS, 3, figsize=(16, 2.5 * N_LANGS))

# Colorbar axes: one per column
cbar_axes = [fig.add_axes([0.92 + 0.025 * c, 0.11, 0.01, 0.77]) for c in range(1)]

for col, (arr, col_title) in enumerate(zip(arrays, col_titles)):
    for lang_idx in range(N_LANGS):
        ax = axes[lang_idx, col]
        data = arr[lang_idx, :].reshape(1, 32)

        # Only draw the colorbar once — on the last column, last row.
        draw_cbar = (col == 2 and lang_idx == N_LANGS - 1)
        sns.heatmap(
            data, ax=ax, cmap="RdBu_r", center=0, vmin=-0.05, vmax=vmax,
            cbar=draw_cbar,
            cbar_ax=cbar_axes[0] if draw_cbar else None,
            xticklabels=[str(i) if i % 8 == 0 else "" for i in range(32)],
            yticklabels=[LANG_NAMES[LANGUAGES[lang_idx]]],
        )
        ax.set_xlabel("Layer" if lang_idx == 3 else "")
        if lang_idx == 0:
            ax.set_title(col_title)

fig.tight_layout(rect=[0, 0, 0.91, 1])

# ── Draw boxes on leftmost column only ────────────────────────────────────
for lang_idx in range(N_LANGS):
    ax = axes[lang_idx, 0]
    ylim   = ax.get_ylim()
    ymin   = min(ylim)
    ymax   = max(ylim)
    rect_y = ymin - 0.01
    rect_h = (ymax - ymin) + 0.02

    # Box 1 — chosen pivots (same x-range for all rows)
    ax.add_patch(Rectangle(
        (BOX_CHOSEN[0], rect_y), BOX_CHOSEN[1] - BOX_CHOSEN[0], rect_h,
        fill=False, edgecolor=COLOR_CHOSEN, linewidth=2,
        linestyle="--", zorder=5, clip_on=False,
    ))

    # Box 2 — language-specific pivots
    x0, x1 = BOX_LANG[lang_idx]
    ax.add_patch(Rectangle(
        (x0, rect_y), x1 - x0, rect_h,
        fill=False, edgecolor=COLOR_LANG, linewidth=2,
        linestyle=(0, (3, 1, 1, 1)),   # dash-dot — distinct from box 1
        zorder=5, clip_on=False,
    ))

# ── Figure-level legend ────────────────────────────────────────────────────
legend_handles = [
    Patch(facecolor="none", edgecolor=COLOR_CHOSEN, linewidth=2,
          linestyle="--", label="Chosen pivots"),
    Patch(facecolor="none", edgecolor=COLOR_LANG,   linewidth=2,
          linestyle=(0, (3, 1, 1, 1)), label="Language-specific pivots"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.45, -0.04),
           frameon=True, framealpha=0.5, facecolor="lightgrey")

plt.savefig("../outputs/figures/fig2_patching_heatmap.pdf", bbox_inches="tight")
plt.savefig("../outputs/figures/fig2_patching_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig2_patching_heatmap.pdf/png")