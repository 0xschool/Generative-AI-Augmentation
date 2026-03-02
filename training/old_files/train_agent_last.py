"""
Step 2 — Train the VAE-DQN agent using official ChefsHatRoomLocal.
Usage: python training/train_agent.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from ChefsHatGym.env.ChefsHatEnv import GAMETYPE
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from agents.vae_dqn_agent_last import VaeDqnAgent

NUM_GAMES        = 10
MATCHES_PER_GAME = 300
VAE_PATH         = "results/models/vae_pretrained.pt"
MODEL_DIR        = "results/models"
LOG_DIR          = "results/logs"
PLOT_DIR         = "results/plots"
METRICS_PATH     = "results/training_metrics.json"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("  Train Agent | Variant 6 (Generative AI Augmentation)")
    print("  Student ID: 16735347")
    print(f"  {NUM_GAMES} games x {MATCHES_PER_GAME} matches = "
          f"{NUM_GAMES*MATCHES_PER_GAME} total matches")
    print("=" * 60)

    vae_path = VAE_PATH if os.path.exists(VAE_PATH) else None
    if vae_path:
        print(f"\n[Info] Using pre-trained VAE: {VAE_PATH}")
    else:
        print("\n[Warning] No VAE found — run pretrain_vae.py first!")

    agent = VaeDqnAgent(
        name              = "DQN_Agent",
        this_agent_folder = MODEL_DIR,
        verbose_console   = False,
        verbose_log       = False,
        log_directory     = LOG_DIR,
        vae_path          = vae_path,
        training          = True,
        latent_dim        = 32,
        lr                = 1e-4,
        gamma             = 0.99,
        epsilon           = 1.0,
        epsilon_min       = 0.05,
        epsilon_decay     = 0.999,
        batch_size        = 64,
        buffer_capacity   = 100_000,
        target_update_freq = 100,
        use_intrinsic_reward = True,
        intrinsic_scale   = 0.05,
    )

    for g in range(NUM_GAMES):
        print(f"\n[Game {g+1}/{NUM_GAMES}] ε={agent.epsilon:.3f} | "
              f"buf={len(agent.buffer):,} | steps={agent.total_steps:,}")

        room = ChefsHatRoomLocal(
            room_name           = f"train_game_{g}",
            game_type           = GAMETYPE["MATCHES"],
            stop_criteria       = MATCHES_PER_GAME,
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
        tot   = sum(wc.values())
        recent = agent.match_rewards[-MATCHES_PER_GAME:] if agent.match_rewards else [0]
        print(f"  ✓ Chef={wc['chef']}({100*wc['chef']/max(tot,1):.1f}%) | "
              f"Dish={wc['dishwasher']}({100*wc['dishwasher']/max(tot,1):.1f}%) | "
              f"AvgR={np.mean(recent):.3f}")

        agent.save_model(tag=f"game_{g}")

    agent.save_model(tag="final")

    # Plots
    def sm(a, w=100):
        return np.convolve(a, np.ones(w)/w, mode="valid").tolist() if len(a)>=w else a

    m   = agent.get_metrics()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training | Student 16735347 | Variant 6", fontsize=12)

    ax = axes[0,0]
    ax.plot(m["match_rewards"], alpha=0.2, color="blue")
    if len(m["match_rewards"]) > 10:
        ax.plot(sm(m["match_rewards"]), color="blue", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title("Match Reward"); ax.set_xlabel("Match"); ax.grid(alpha=0.3)

    ax = axes[0,1]
    ax.axhline(25, color="red", linestyle="--", label="Random (25%)")
    ax.set_title("Chef Rate"); ax.set_ylim(0,100); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1,0]
    if m["losses"]:
        ax.plot(m["losses"], alpha=0.2, color="red")
        if len(m["losses"]) > 50:
            ax.plot(sm(m["losses"], 200), color="red", lw=1.5)
    ax.set_title("DQN Loss"); ax.set_xlabel("Step"); ax.grid(alpha=0.3)

    ax   = axes[1,1]
    wc   = m["win_counts"]
    tot  = sum(wc.values())
    cnts = list(wc.values())
    cols = ["gold","silver","#cd7f32","#888"]
    bars = ax.bar(list(wc.keys()), cnts, color=cols, edgecolor="black")
    for b, c in zip(bars, cnts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f"{c}\n({100*c/max(tot,1):.0f}%)", ha="center", fontsize=9)
    ax.axhline(tot*0.25, color="red", linestyle="--", label="25%")
    ax.set_title("Position Distribution"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/training_results.png", dpi=150)
    plt.close()

    out = {"student_id":"16735347","variant":6,
           "total_matches":sum(m["win_counts"].values()),
           "total_steps":m["total_steps"],"final_epsilon":m["final_epsilon"],
           "win_counts":m["win_counts"],
           "chef_rate":m["win_counts"]["chef"]/max(sum(m["win_counts"].values()),1),
           "timestamp":datetime.now().isoformat()}
    json.dump(out, open(METRICS_PATH,"w"), indent=2)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    wc  = m["win_counts"]
    tot = sum(wc.values())
    for role, cnt in wc.items():
        print(f"  {role:12s}: {cnt:5d}  ({100*cnt/max(tot,1):.1f}%)")
    print(f"  Chef rate : {100*wc['chef']/max(tot,1):.1f}%  (~25% random)")
    print(f"  Steps     : {m['total_steps']:,}")
    print("=" * 60)
    print("[Next] python evaluation/evaluate_agent.py")
