"""
Step 2 — Train the VAE-DQN agent.

Uses the OFFICIAL ChefsHatRoomLocal from the cloned repo.
Our agent sits alongside the official AgentRandon opponents.

Usage (from project root):
    python training/train_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from ChefsHatGym.env.ChefsHatEnv import GAMETYPE
from ChefsHatGym.agents.agent_random import AgentRandon       # official random agent
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal  # official room

from agents.vae_dqn_agent import VaeDqnAgent

# ── Config ───────────────────────────────────────────────────────────────────

NUM_GAMES        = 20    
MATCHES_PER_GAME = 500   

VAE_PATH         = "results/models/vae_pretrained.pt"
MODEL_DIR        = "results/models"
LOG_DIR          = "results/logs"
PLOT_DIR         = "results/plots"
METRICS_PATH     = "results/training_metrics.json"

AGENT_CONFIG = dict(
    latent_dim          = 32,
    lr                  = 1e-4,
    gamma               = 0.99,
    epsilon             = 1.0,
    epsilon_min         = 0.05,
    epsilon_decay       = 0.99995,
    batch_size          = 64,
    buffer_capacity     = 100_000,
    target_update_freq  = 500,
    use_intrinsic_reward = True,
    intrinsic_scale     = 0.05,

)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


def run_game(agent, game_idx, matches):
    """Run one training game using the official ChefsHatRoomLocal."""
    room = ChefsHatRoomLocal(
        room_name           = f"train_{game_idx}",
        game_type           = GAMETYPE["MATCHES"],
        stop_criteria       = matches,
        max_rounds          = -1,
        save_dataset        = True,
        verbose_console     = False,
        verbose_log         = False,
        game_verbose_console = False,
        game_verbose_log    = True,
        log_directory       = LOG_DIR,
    )

    # Official AgentRandon as opponents
    opponents = [
        AgentRandon(
            name           = f"Random_{i}",
            verbose_console = False,
            verbose_log    = False,
            log_directory  = LOG_DIR,
        )
        for i in range(3)
    ]

    # Add players in order: our agent first, then 3 randoms
    for p in [agent] + opponents:
        room.add_player(p)

    print(f"\n[Train] Game {game_idx+1}/{NUM_GAMES} | "
          f"{matches} matches | ε={agent.epsilon:.4f} | "
          f"buffer={len(agent.buffer)}")

    room.start_new_game()

    wc    = agent.win_counts
    total = sum(wc.values())
    avg_r = float(np.mean(agent.match_rewards[-matches:])) if agent.match_rewards else 0.0
    chef_pct = 100 * wc["chef"] / max(total, 1)

    print(f"  ✓ Chef: {wc['chef']} ({chef_pct:.1f}%) | "
          f"Avg reward (last {matches}): {avg_r:.3f} | "
          f"Steps: {agent.total_steps}")

    return {"game": game_idx, "avg_match_reward": avg_r,
            "win_counts": dict(wc), "epsilon": agent.epsilon}


def smooth(arr, w=50):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid").tolist()


def plot_results(agent):
    m   = agent.get_metrics()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "VAE-DQN Training | Student 16735347 | Variant 6 (Generative AI Augmentation)",
        fontsize=11,
    )

    # Episode rewards
    ax = axes[0, 0]
    r  = m["episode_rewards"]
    if r:
        ax.plot(r, alpha=0.25, color="blue")
        ax.plot(smooth(r), color="blue", lw=1.5, label="Smoothed")
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Match terminal rewards
    ax = axes[0, 1]
    mr = m["match_rewards"]
    if mr:
        ax.plot(mr, alpha=0.25, color="green")
        ax.plot(smooth(mr), color="green", lw=1.5)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_title("Match Terminal Reward")
        ax.set_xlabel("Match")
        ax.grid(True, alpha=0.3)

    # DQN loss
    ax = axes[1, 0]
    ls = m["losses"]
    if ls:
        ax.plot(ls, alpha=0.2, color="red")
        ax.plot(smooth(ls, 200), color="red", lw=1.5)
        ax.set_title("DQN Huber Loss")
        ax.set_xlabel("Training Step")
        ax.grid(True, alpha=0.3)

    # Win count bars
    ax    = axes[1, 1]
    wc    = m["win_counts"]
    roles = list(wc.keys())
    cnts  = [wc[r] for r in roles]
    cols  = ["gold", "silver", "#cd7f32", "#888"]
    bars  = ax.bar(roles, cnts, color=cols, edgecolor="black")
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                str(c), ha="center", va="bottom")
    ax.set_title("Finishing Position Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  DQN Training | Variant 6 (Generative AI Augmentation)")
    print("  Student ID: 16735347")
    print("=" * 60)

    if not os.path.exists(VAE_PATH):
        print(f"\n[WARNING] No VAE found at {VAE_PATH}.")
        print("[WARNING] Run pretrain_vae.py first for best results.")
        vae_path = None
    else:
        print(f"\n[INFO] Using pre-trained VAE: {VAE_PATH}")
        vae_path = VAE_PATH

    agent = VaeDqnAgent(
        name                = "DQN_Agent",
        this_agent_folder   = MODEL_DIR,
        verbose_console     = False,
        verbose_log         = True,
        log_directory       = LOG_DIR,
        vae_path            = vae_path,
        training            = True,
        **AGENT_CONFIG,
    )

    game_logs = []
    for g in range(NUM_GAMES):
        log = run_game(agent, g, MATCHES_PER_GAME)
        game_logs.append(log)
        agent.save_model(tag=f"game_{g}")

    agent.save_model(tag="final")
    plot_results(agent)

    m = agent.get_metrics()
    output = {
        "student_id": "16735347",
        "variant": 6,
        "config": AGENT_CONFIG,
        "total_steps": m["total_steps"],
        "final_epsilon": m["final_epsilon"],
        "win_counts": m["win_counts"],
        "avg_reward_overall": float(np.mean(m["match_rewards"])) if m["match_rewards"] else 0,
        "per_game": game_logs,
        "timestamp": datetime.now().isoformat(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Metrics] Saved → {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    wc    = m["win_counts"]
    total = sum(wc.values())
    for role, cnt in wc.items():
        pct = 100 * cnt / max(total, 1)
        print(f"  {role:12s}: {cnt:4d}  ({pct:.1f}%)")
    print(f"  Total steps : {m['total_steps']:,}")
    print(f"  Final ε     : {m['final_epsilon']:.4f}")
    print("=" * 60)
    print("[Next] Run: python evaluation/evaluate_agent.py")
