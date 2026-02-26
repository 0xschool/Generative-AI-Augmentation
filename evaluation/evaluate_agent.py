"""
Step 3 — Evaluate the trained agent.

Tests the agent against 3 official AgentRandon opponents across multiple seeds.
Also visualises the VAE latent space via PCA.

Usage (from project root):
    python evaluation/evaluate_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym
from ChefsHatGym.env.ChefsHatEnv import GAMETYPE
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal

from agents.vae_dqn_agent import VaeDqnAgent
from models.vae import VAE

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR   = "results/models"
VAE_PATH    = "results/models/vae_pretrained.pt"
LOG_DIR     = "results/logs"
PLOT_DIR    = "results/plots"
RESULTS_OUT = "results/evaluation_results.json"

EVAL_MATCHES = 100
EVAL_SEEDS   = 3
ACTION_DIM   = 200
OBS_DIM      = 228

os.makedirs(PLOT_DIR, exist_ok=True)


def load_agent(seed):
    np.random.seed(seed)
    agent = VaeDqnAgent(
        name              = f"Eval_Agent_s{seed}",
        this_agent_folder = MODEL_DIR,
        verbose_console   = False,
        verbose_log       = False,
        log_directory     = LOG_DIR,
        vae_path          = VAE_PATH if os.path.exists(VAE_PATH) else None,
        training          = False,
        epsilon           = 0.05,
    )
    agent.load_model(tag="final")
    agent.win_counts   = {k: 0 for k in agent.win_counts}
    agent.match_rewards = []
    return agent


def evaluate_seed(seed):
    agent = load_agent(seed)

    room = ChefsHatRoomLocal(
        room_name           = f"eval_s{seed}",
        game_type           = GAMETYPE["MATCHES"],
        stop_criteria       = EVAL_MATCHES,
        max_rounds          = -1,
        save_dataset        = True,
        verbose_console     = False,
        verbose_log         = False,
        game_verbose_console = False,
        game_verbose_log    = False,
        log_directory       = LOG_DIR,
    )
    for p in [agent] + [
        AgentRandon(name=f"Rand_{i}", verbose_console=False,
                    verbose_log=False, log_directory=LOG_DIR)
        for i in range(3)
    ]:
        room.add_player(p)

    room.start_new_game()

    wc    = agent.win_counts
    total = sum(wc.values())
    return {
        "seed":             seed,
        "win_counts":       dict(wc),
        "chef_rate":        wc["chef"] / max(total, 1),
        "avg_match_reward": float(np.mean(agent.match_rewards)) if agent.match_rewards else 0.0,
    }


def latent_space_analysis():
    """PCA of VAE latent codes — shows how well game states are organised."""
    if not os.path.exists(VAE_PATH):
        print("[Latent] No VAE found — skipping.")
        return

    from sklearn.decomposition import PCA

    vae = VAE(OBS_DIM, 128, 32)
    vae.load(VAE_PATH)
    vae.eval()

    obs_list, labels = [], []
    try:
        env = gym.make("chefshat-v1")
        from ChefsHatGym.env import ChefsHatEnv
        env.startExperiment(
            gameType=GAMETYPE["MATCHES"], stopCriteria=3, maxRounds=-1,
            playerNames=["P0","P1","P2","P3"], logDirectory=LOG_DIR,
            verbose=False, saveDataset=False, saveLog=False,
        )
        env.reset()
        steps = 0
        while not env.gameFinished and len(obs_list) < 500 and steps < 5000:
            obs = env.getObservation()
            obs_list.append(obs.copy())
            n_valid = int(np.sum(obs[28:]))
            labels.append(min(n_valid // 5, 9))

            vi = np.where(obs[28:] == 1)[0]
            if len(vi) == 0:
                vi = np.array([ACTION_DIM - 1])
            a = np.zeros(ACTION_DIM, dtype=np.float32)
            a[int(np.random.choice(vi))] = 1

            info = {"Action_Valid": False}
            att  = 0
            while not info["Action_Valid"] and att < 10:
                _, _, mo, _, info = env.step(a)
                att += 1
            if mo and not env.gameFinished:
                try:
                    pd, dc, pw, wc, ps, sc, pc, cc = env.get_chef_souschef_roles_cards()
                    env.exchange_cards(cc[:2] if len(cc)>=2 else cc, sc[0] if sc else [], wc, dc, False, -1)
                except Exception:
                    pass
            steps += 1
        env.close()
    except Exception as e:
        print(f"[Latent] Env error: {e}")
        return

    if len(obs_list) < 10:
        return

    data_t = torch.FloatTensor(np.array(obs_list))
    with torch.no_grad():
        latents = vae.get_latent(data_t).numpy()
        recons  = vae.decode(torch.FloatTensor(latents)).numpy()
    mse_arr = np.mean((recons - np.array(obs_list))**2, axis=1)

    pca     = PCA(n_components=2)
    reduced = pca.fit_transform(latents)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VAE Latent Space Analysis", fontsize=13)

    ax = axes[0]
    sc = ax.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10", s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Valid-action bucket")
    ax.set_title(f"PCA of Latent Space (n={len(obs_list)}, 32→2 dims)")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(mse_arr, bins=40, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(mse_arr), color="red", linestyle="--",
               label=f"Mean: {np.mean(mse_arr):.5f}")
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("MSE per observation")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "latent_space_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Latent] Plot saved → {path}")


def plot_eval_summary(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation: VAE-DQN vs 3 Random Agents", fontsize=12)

    seeds     = [r["seed"] for r in results]
    chef_pcts = [r["chef_rate"] * 100 for r in results]
    avg_rs    = [r["avg_match_reward"] for r in results]

    ax = axes[0]
    bars = ax.bar([f"Seed {s}" for s in seeds], chef_pcts, color="gold", edgecolor="black")
    ax.axhline(25, color="red", linestyle="--", label="Random baseline (25%)")
    for bar, v in zip(bars, chef_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom")
    ax.set_title("Chef (1st) Win Rate per Seed")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar([f"Seed {s}" for s in seeds], avg_rs, color="green", edgecolor="black")
    ax.axhline(0, color="black", linestyle="--")
    for bar, v in zip(bars, avg_rs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom")
    ax.set_title("Average Match Reward per Seed")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "evaluation_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Summary plot → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Evaluation | Variant 6 (Generative AI Augmentation)")
    print("  Student ID: 16735347")
    print("=" * 60)

    print("\n[1/3] VAE latent space analysis...")
    latent_space_analysis()

    print("\n[2/3] Agent evaluation vs random opponents...")
    results = []
    for seed in range(EVAL_SEEDS):
        print(f"  Seed {seed}: {EVAL_MATCHES} matches...")
        r = evaluate_seed(seed)
        results.append(r)
        wc = r["win_counts"]
        print(f"    Chef={wc['chef']} | Sous={wc['sous_chef']} | "
              f"Waiter={wc['waiter']} | Dish={wc['dishwasher']} | "
              f"Chef rate={r['chef_rate']*100:.1f}%")

    print("\n[3/3] Plotting...")
    plot_eval_summary(results)

    output = {
        "student_id": "16735347",
        "variant": 6,
        "eval_seeds": EVAL_SEEDS,
        "eval_matches": EVAL_MATCHES,
        "results": results,
        "mean_chef_rate": float(np.mean([r["chef_rate"] for r in results])),
        "std_chef_rate":  float(np.std( [r["chef_rate"] for r in results])),
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Done] Results → {RESULTS_OUT}")
    print("=" * 60)
    print(f"  Mean Chef Rate: {output['mean_chef_rate']*100:.1f}% "
          f"± {output['std_chef_rate']*100:.1f}%")
    print(f"  Random baseline: ~25%")
    print("=" * 60)
