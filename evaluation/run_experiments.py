"""
Step 4 — Ablation experiments comparing agent configurations.

Compares: with/without VAE curiosity, different latent dimensions.

Usage (from project root):
    python evaluation/run_experiments.py
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

from ChefsHatGym.env.ChefsHatEnv import GAMETYPE
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal

from agents.vae_dqn_agent import VaeDqnAgent

LOG_DIR  = "results/logs"
PLOT_DIR = "results/plots"
VAE_PATH = "results/models/vae_pretrained.pt"
TRAIN_M  = 300   # matches per experiment (short for comparison)
EVAL_M   = 100


os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

EXPERIMENTS = [
    {"name": "VAE + Curiosity (latent=32)", "color": "blue",
     "cfg": {"latent_dim": 32, "use_intrinsic_reward": True,  "intrinsic_scale": 0.05}},
    {"name": "VAE, No Curiosity (latent=32)", "color": "green",
     "cfg": {"latent_dim": 32, "use_intrinsic_reward": False}},
    {"name": "VAE + Curiosity (latent=16)", "color": "orange",
     "cfg": {"latent_dim": 16, "use_intrinsic_reward": True,  "intrinsic_scale": 0.05}},
    {"name": "VAE + Curiosity (latent=64)", "color": "purple",
     "cfg": {"latent_dim": 64, "use_intrinsic_reward": True,  "intrinsic_scale": 0.05}},
]


def run_experiment(exp, idx):
    name = exp["name"]
    cfg  = exp["cfg"]
    vae  = VAE_PATH if os.path.exists(VAE_PATH) else None
    print(f"\n[Exp {idx+1}/{len(EXPERIMENTS)}] {name}")

    # Train
    agent = VaeDqnAgent(
        name              = f"Exp{idx}",
        this_agent_folder = f"results/models/exp_{idx}",
        verbose_console   = False,
        verbose_log       = False,
        log_directory     = LOG_DIR,
        vae_path          = vae,
        training          = True,
        epsilon           = 1.0, epsilon_min=0.05, epsilon_decay=0.9995,
        gamma=0.99, lr=5e-4, batch_size=64,
        buffer_capacity=20_000, target_update_freq=300,
        **cfg,
    )

    room = ChefsHatRoomLocal(
        room_name=f"exp_{idx}_train", game_type=GAMETYPE["MATCHES"],
        stop_criteria=TRAIN_M, max_rounds=-1, save_dataset=False,
        verbose_console=False, verbose_log=False,
        game_verbose_console=False, game_verbose_log=False,
        log_directory=LOG_DIR,
    )
    for p in [agent] + [AgentRandon(name=f"T{i}", verbose_console=False,
              verbose_log=False, log_directory=LOG_DIR) for i in range(3)]:
        room.add_player(p)
    room.start_new_game()
    train_rewards = list(agent.match_rewards)

    # Eval
    agent.training  = False
    agent.epsilon   = 0.05
    agent.win_counts = {k: 0 for k in agent.win_counts}
    agent.match_rewards = []

    eval_room = ChefsHatRoomLocal(
        room_name=f"exp_{idx}_eval", game_type=GAMETYPE["MATCHES"],
        stop_criteria=EVAL_M, max_rounds=-1, save_dataset=False,
        verbose_console=False, verbose_log=False,
        game_verbose_console=False, game_verbose_log=False,
        log_directory=LOG_DIR,
    )
    for p in [agent] + [AgentRandon(name=f"E{i}", verbose_console=False,
              verbose_log=False, log_directory=LOG_DIR) for i in range(3)]:
        eval_room.add_player(p)
    eval_room.start_new_game()

    wc    = agent.win_counts
    total = sum(wc.values())
    chef_rate = wc["chef"] / max(total, 1)
    print(f"  Chef rate: {chef_rate*100:.1f}% | counts: {dict(wc)}")

    return {"name": name, "color": exp["color"], "cfg": cfg,
            "train_rewards": train_rewards,
            "eval_win_counts": dict(wc),
            "eval_chef_rate": chef_rate}


def plot_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Architecture Ablation | Variant 6: Generative AI Augmentation", fontsize=12)

    ax = axes[0]
    for r in results:
        tr = r["train_rewards"]
        if tr:
            w = min(20, max(1, len(tr)//3))
            s = np.convolve(tr, np.ones(w)/w, mode="valid")
            ax.plot(s, label=r["name"], color=r["color"], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title("Training Match Reward (Smoothed)")
    ax.set_xlabel("Match")
    ax.set_ylabel("Terminal Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = [r["name"] for r in results]
    pcts  = [r["eval_chef_rate"]*100 for r in results]
    cols  = [r["color"] for r in results]
    bars  = ax.bar(range(len(names)), pcts, color=cols, edgecolor="black")
    ax.axhline(25, color="red", linestyle="--", label="Random (25%)")
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_title(f"Chef Rate ({EVAL_M} eval matches)")
    ax.set_ylabel("Chef Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "experiment_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Experiments] Plot → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Ablation Experiments | Variant 6 (Generative AI Aug)")
    print("  Student ID: 16735347")
    print("=" * 60)

    all_results = []
    for i, exp in enumerate(EXPERIMENTS):
        r = run_experiment(exp, i)
        all_results.append(r)

    plot_comparison(all_results)

    best = max(all_results, key=lambda r: r["eval_chef_rate"])
    print(f"\n[Best] {best['name']} → Chef rate {best['eval_chef_rate']*100:.1f}%")

    out = [{"name": r["name"], "cfg": r["cfg"],
            "eval_win_counts": r["eval_win_counts"],
            "eval_chef_rate": r["eval_chef_rate"]}
           for r in all_results]
    with open("results/experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[Done] results/experiment_results.json saved")
