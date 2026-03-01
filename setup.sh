#!/bin/bash
# =============================================================================
# setup.sh — Clone official ChefsHatGYM repo and set up environment
#
# Run this ONCE before anything else:
#     bash setup.sh
# =============================================================================

set -e   # stop on any error

echo "============================================================"
echo "  Chef's Hat RL Setup | Student ID: 16735347 | Variant 6"
echo "============================================================"

# ── 1. Clone official ChefsHatGYM repo into this folder ──────────────────
if [ -d "ChefsHatGYM" ]; then
    echo "[1/4] ChefsHatGYM already cloned. Pulling latest..."
    cd ChefsHatGYM && git pull && cd ..
else
    echo "[1/4] Cloning official ChefsHatGYM repo..."
    git clone https://github.com/pablovin/ChefsHatGYM.git
fi

# ── 2. Install ChefsHatGYM dependencies ──────────────────────────────────
echo "[2/4] Installing ChefsHatGYM requirements..."
pip install -r ChefsHatGYM/Requirements.txt

# ── 3. Install ChefsHatGYM as a package (from cloned src) ────────────────
echo "[3/4] Installing ChefsHatGYM package from source..."
pip install -e ChefsHatGYM/

# ── 4. Install our project dependencies ──────────────────────────────────
echo "[4/4] Installing project dependencies..."
pip install torch numpy matplotlib scikit-learn
pip install chefshatgym
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.24.3" --force-reinstall
pip install gym


# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Now run the pipeline in order:"
echo "    python training/pretrain_vae.py    # Step 1: train VAE"
echo "    python training/train_agent.py     # Step 2: train DQN"
echo "    python evaluation/evaluate_agent.py # Step 3: evaluate"
echo "    python evaluation/run_experiments.py # Step 4: ablations"
echo "============================================================"
