"""
Step 1 — Pre-train the VAE on game state observations.

Runs random agents through the official ChefsHatRoomLocal to collect
228-dim observations, then trains a VAE to compress them to 32 dims.

Usage (from project root):
    python training/pretrain_vae.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.env.ChefsHatEnv import GAMETYPE
from ChefsHatGym.agents.agent_random import AgentRandon   # official random agent
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal

from models.vae import VAE

# ── Config ───────────────────────────────────────────────────────────────────
NUM_GAMES        = 20       # random-play games to collect observations from
MATCHES_PER_GAME = 50       # matches per collection game
VAE_EPOCHS       = 100
BATCH_SIZE       = 256
LR               = 1e-3
LATENT_DIM       = 32
BETA             = 1.0      # KL weight

VAE_SAVE_PATH  = "results/models/vae_pretrained.pt"
PLOT_SAVE_PATH = "results/plots/vae_training_loss.png"
LOG_DIR        = "results/logs"
OBS_DIM        = 228
ACTION_DIM     = 200

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/plots",  exist_ok=True)
os.makedirs(LOG_DIR,           exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect observations via random play
# ─────────────────────────────────────────────────────────────────────────────

def collect_observations(num_games=20, matches=50):
    print(f"\n[VAE] Collecting observations: {num_games} games × {matches} matches")
    all_obs = []

    for g in range(num_games):
        try:
            # Use official env directly for observation collection
            env = gym.make("chefshat-v1")
            env.startExperiment(
                gameType=GAMETYPE["MATCHES"],
                stopCriteria=matches,
                maxRounds=-1,
                playerNames=["R0", "R1", "R2", "R3"],
                logDirectory=LOG_DIR,
                verbose=False,
                saveDataset=False,
                saveLog=False,
            )
            env.reset()

            steps = 0
            while not env.gameFinished and steps < 20_000:
                obs = env.getObservation()
                all_obs.append(obs.copy())

                # Pick a random valid action (same logic as official AgentRandon)
                possible = obs[28:]
                valid    = np.where(possible == 1)[0]
                if len(valid) == 0:
                    valid = np.array([ACTION_DIM - 1])
                idx    = int(np.random.choice(valid))
                action = np.zeros(ACTION_DIM, dtype=np.float32)
                action[idx] = 1

                info = {"Action_Valid": False}
                attempts = 0
                while not info["Action_Valid"] and attempts < 20:
                    _, _, match_over, _, info = env.step(action)
                    attempts += 1
                    if not info["Action_Valid"]:
                        obs2  = env.getObservation()
                        v2    = np.where(obs2[28:] == 1)[0]
                        if len(v2) > 0:
                            idx    = int(np.random.choice(v2))
                            action = np.zeros(ACTION_DIM, dtype=np.float32)
                            action[idx] = 1

                if match_over and not env.gameFinished:
                    try:
                        pd, dc, pw, wc, ps, sc, pc, cc = env.get_chef_souschef_roles_cards()
                        sc_card    = sc[0] if sc else []
                        chef_cards = cc[:2] if len(cc) >= 2 else cc
                        env.exchange_cards(chef_cards, sc_card, wc, dc, False, -1)
                    except Exception:
                        pass

                steps += 1

            env.close()

        except Exception as e:
            print(f"  [Game {g}] Error: {e}")
            continue

        if (g + 1) % 5 == 0:
            print(f"  Collected {len(all_obs):,} obs from {g+1} games")

    print(f"\n[VAE] Total observations: {len(all_obs):,}")
    return np.array(all_obs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train VAE
# ─────────────────────────────────────────────────────────────────────────────

def train_vae(observations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[VAE] Training on {len(observations):,} samples | device={device}")

    vae       = VAE(OBS_DIM, 128, LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    data      = torch.FloatTensor(observations)
    n         = len(data)

    total_hist, recon_hist, kl_hist = [], [], []

    vae.train()
    for epoch in range(VAE_EPOCHS):
        perm = torch.randperm(n)
        data = data[perm]
        e_total = e_recon = e_kl = 0.0
        n_batch = 0

        for start in range(0, n, BATCH_SIZE):
            batch   = data[start:start + BATCH_SIZE].to(device)
            recon, mu, logvar = vae(batch)
            loss, rl, kl = VAE.loss(recon, batch, mu, logvar, BETA)
            loss = loss / len(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            optimizer.step()

            e_total += loss.item()
            e_recon += rl.item() / len(batch)
            e_kl    += kl.item() / len(batch)
            n_batch += 1

        scheduler.step()
        total_hist.append(e_total / n_batch)
        recon_hist.append(e_recon / n_batch)
        kl_hist.append(e_kl    / n_batch)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{VAE_EPOCHS} | "
                  f"Total: {total_hist[-1]:.4f} | "
                  f"Recon: {recon_hist[-1]:.4f} | "
                  f"KL: {kl_hist[-1]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("VAE Pre-Training Loss", fontsize=13)
    for ax, hist, label, colour in zip(
        axes,
        [total_hist, recon_hist, kl_hist],
        ["Total Loss", "Reconstruction Loss", "KL Divergence"],
        ["purple", "blue", "orange"],
    ):
        ax.plot(hist, color=colour)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()
    print(f"[VAE] Training plot → {PLOT_SAVE_PATH}")

    vae.eval()
    with torch.no_grad():
        sample = data[:256].to(device)
        recon, mu, _ = vae(sample)
        mse  = torch.mean((recon - sample) ** 2).item()
        norm = mu.norm(dim=1).mean().item()
    print(f"[VAE Eval] Reconstruction MSE : {mse:.6f}")
    print(f"[VAE Eval] Latent mean norm   : {norm:.4f}")

    return vae


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VAE Pre-Training | Variant 6 (Generative AI Augmentation)")
    print("  Student ID: 16735347")
    print("=" * 60)

    obs = collect_observations(NUM_GAMES, MATCHES_PER_GAME)
    if len(obs) == 0:
        print("[ERROR] No observations collected.")
        sys.exit(1)

    vae = train_vae(obs)
    vae.save(VAE_SAVE_PATH)

    print(f"\n[Done] VAE saved → {VAE_SAVE_PATH}")
    print("[Next] Run: python training/train_agent.py")
