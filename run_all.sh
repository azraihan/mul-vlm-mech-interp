#!/bin/bash
set -euo pipefail    # exit on error, unset variable, or pipe failure

# Always run from the directory containing this script,
# so all relative paths (data/, outputs/, etc.) resolve correctly.
cd "$(dirname "$0")"

echo "[1/9] Installing dependencies..."
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 2>/dev/null
pip install -r setup/requirements.txt -q

echo "[2/9] Downloading and preparing data..."
python data/build_probing_set.py
python data/download_xgqa.py
python data/download_mmmb.py

echo "[3/9] Experiment 1: Logit Lens..."
python experiments/exp1_logit_lens.py

echo "[4/9] Experiment 2: Activation Patching (writes pivot_layers.json)..."
python experiments/exp2_activation_patching.py

echo "[5/9] Experiment 3: Steering Vectors + Evaluation..."
python experiments/exp3_steering.py

echo "[6/9] Experiment 4: Ablations..."
python experiments/exp4_ablations.py

echo "[7/9] Generating figures..."
python figures/plot_fig1_logit_lens.py
python figures/plot_fig2_patching_heatmap.py
python figures/plot_fig3_visual_contribution.py
python figures/plot_fig4_alpha_curves.py
python figures/plot_fig5_qualitative.py

echo "[8/9] Generating tables..."
python tables/generate_tables.py > tables/tables.tex

echo "[9/9] Verifying outputs..."
python - <<'EOF'
import os, sys
required = [
    "outputs/logit_lens/logit_lens_mean_en.npy",
    "outputs/logit_lens/logit_lens_mean_tl.npy",
    "outputs/patching/patching_residual.npy",
    "outputs/patching/patching_attn.npy",
    "outputs/patching/patching_mlp.npy",
    "outputs/patching/patching_visual.npy",
    "outputs/pivot_layers.json",
    "outputs/steering_vectors/steering_vec_fr.pt",
    "outputs/steering_vectors/steering_vec_ar.pt",
    "outputs/steering_vectors/steering_vec_zh.pt",
    "outputs/steering_vectors/steering_vec_bn.pt",
    "outputs/ablations/layer_range_ablation.json",
    "outputs/ablations/visual_token_ablation.json",
    "outputs/ablations/alpha_sensitivity.json",
    "outputs/ablations/cross_lingual_transfer.json",
    "outputs/figures/fig1_logit_lens.pdf",
    "outputs/figures/fig2_patching_heatmap.pdf",
    "outputs/figures/fig3_visual_contribution.pdf",
    "outputs/figures/fig4_alpha_curves.pdf",
    "outputs/figures/fig5_qualitative.pdf",
    "tables/tables.tex",
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("MISSING FILES:")
    for f in missing: print(f"  {f}")
    sys.exit(1)
print("ALL REQUIRED OUTPUTS PRESENT.")
EOF

echo "=== PIPELINE COMPLETE. Check outputs/figures/ and tables/tables.tex ==="
