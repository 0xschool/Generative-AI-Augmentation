# Chef's Hat RL — Variant 6: Generative AI Augmentation

**Student ID:** 16735347  
**Module:** Reinforcement Learning  
**Environment:** [ChefsHatGYM (Official)](https://github.com/pablovin/ChefsHatGYM)

---

## Overview

This project implements a **VAE-Augmented Dueling Double DQN** agent for the Chef's Hat card game.

The Generative AI component is a **Variational Autoencoder (VAE)** that:
1. Compresses 228-dim game observations → 32-dim latent codes (feature extraction)
2. Provides a curiosity-based intrinsic reward via reconstruction error (exploration signal)

---

## Requirements

- Python 3.9 or 3.10 (recommended)
- pip
- git
- ~2 GB free disk space

---

## Setup Instructions

### Step 1 — Clone this repository
```bash
git clone https://github.com/0xschool/Generative-AI-Augmentation.git
cd Generative-AI-Augmentation
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / Mac
# OR on Windows:
venv\Scripts\activate
```

### Step 3 — Run the setup script
```bash
bash setup.sh
```
### Step 4 — Run the dependency scripts to solve the issues of module versioning if setup.sh fails
```
pip install "numpy==1.24.3" --force-reinstall
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib
pip install gym
pip install chefshatgym

```


---

## Running the Pipeline

All commands must be run from the **project root folder**.

### Step 1 — Pre-train the VAE (~5 minutes)
```bash
python training/pretrain_vae.py
```
Collects ~100k game observations via random play, trains the VAE to compress
228-dim states to 32-dim latent codes. Saves to `results/models/vae_pretrained.pt`.

### Step 2 — Train the DQN Agent (~20-40 minutes)
```bash
python training/train_agent.py
```
Trains a Dueling Double DQN against 3 official `AgentRandon` opponents over
10 games × 300 matches. Prints progress every 100 matches. Saves final model
to `results/models/dqn_final.pt`.

### Step 3 — Evaluate the Agent (~5 minutes)
```bash
python evaluation/evaluate_agent.py
```
Tests the trained agent vs 3 random opponents across 3 seeds (100 matches each).
Generates PCA visualisation of the VAE latent space and a win rate summary.
A Chef rate above 25% means the agent beats the random baseline.

### Step 4 — Ablation Experiments (optional, ~30 minutes)
```bash
python evaluation/run_experiments.py
```
Compares: with/without curiosity, latent dimensions 16 / 32 / 64.

---

## Project Structure

```
chefs_hat_rl/
├── README.md                        <- You are here
├── setup.sh                         <- Run once to install everything
│
├── ChefsHatGYM/                     <- Cloned by setup.sh (official repo)
│
├── agents/
│   └── vae_dqn_agent.py             <- Our DQN agent
│
├── models/
│   ├── vae.py                       <- Variational Autoencoder (Gen AI component)
│   ├── dqn.py                       <- Dueling DQN network
│   └── replay_buffer.py             <- Experience replay
│
├── training/
│   ├── pretrain_vae.py              <- Step 1: train VAE
│   └── train_agent.py               <- Step 2: train DQN
│
├── evaluation/
│   ├── evaluate_agent.py            <- Step 3: evaluate
│   └── run_experiments.py           <- Step 4: ablations (optional)
│
└── results/                         <- Created automatically on first run
    ├── models/                      <- Saved model weights (.pt files)
    ├── plots/                       <- Training and evaluation figures (.png)
    └── logs/                        <- Game logs from ChefsHatGYM
```

---

## Architecture Summary

```
Raw Observation (228-dim)
        |
        v
[ VAE Encoder ]   <- Pre-trained unsupervised, FROZEN during RL
   228 -> 32
        |
        v  latent z (32-dim)
[ Dueling DQN ]   <- Trained via Double DQN + experience replay
   32 -> 200
        |
        v  Q-values masked to valid actions only
    Action chosen

Reward:
  Per step  : -0.002 step penalty + 0.01 x cards played + 0.05 x pizza
  Curiosity : VAE reconstruction error x 0.05 (intrinsic exploration bonus)
  Terminal  : Chef=+1.0 | Sous-Chef=+0.33 | Waiter=-0.33 | Dishwasher=-1.0
```

---

## Common Issues

| Error | Fix |
|---|---|
| numpy version conflict | `pip install numpy==1.24.3` |
| matplotlib ImportError | `pip install --upgrade matplotlib` |
| ModuleNotFoundError: ChefsHatGym | `pip install -e ChefsHatGYM/ --no-deps` |
| VAE size mismatch | Delete `results/models/vae_pretrained.pt` and re-run `pretrain_vae.py` |
| Training very slow | Reduce `NUM_MATCHES` in `train_agent.py` to 500 for a quick test |

---

## AI Usage Declaration

Claude (Anthropic) assisted with code scaffolding and debugging the ChefsHatGYM v3 API.
All design decisions were made by the student (VAE choice, DQN variant,
reward shaping, curiosity signal, two-stage training pipeline).
