"""
VAE-Augmented DQN Agent — Variant 6: Generative AI Augmentation
Student ID: 16735347  |  16735347 mod 7 = 6

Follows the EXACT same interface as the official ChefsHatGYM agent_random.py:
    - Inherits ChefsHatPlayer
    - Implements get_action(observations)
    - Implements get_reward(envInfo)  using envInfo["Author_Index"],
      envInfo["Match_Score"], envInfo["Finished_Players"]
    - Implements get_exhanged_cards(cards, amount)
    - Implements update_my_action, update_action_others, update_end_match,
      update_start_match, do_special_action, update_exchange_cards

Generative AI components:
    VAE encoder  → compresses obs[228] to latent[32]  (feature extractor, frozen)
    VAE decoder  → reconstruction error = curiosity reward  (intrinsic signal)
    Dueling DQN  → maps latent[32] → Q-values[200]
"""

import sys
import os
# Make sure ChefsHatGYM src is on path when running from cloned repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))

import numpy as np
import torch
import torch.optim as optim

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
#from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
from ChefsHatGym.rewards.performance_score import RewardPerformanceScore

# Add project models to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vae          import VAE
from models.dqn          import DuelingDQN
from models.replay_buffer import ReplayBuffer

# ── Constants (from official ChefsHatGYM source) ─────────────────────────────
OBS_DIM    = 228
BOARD_DIM  = 11    # obs[0:11]
HAND_DIM   = 17    # obs[11:28]
ACTION_DIM = 200   # obs[28:228]


class VaeDqnAgent(ChefsHatPlayer):
    """
    Dueling Double DQN agent with VAE-encoded state representations.

    The VAE is pre-trained unsupervised on game observations, then its encoder
    is frozen and used to compress raw 228-dim states into 32-dim latent codes.
    The DQN policy network operates on these codes.

    A curiosity-based intrinsic reward (VAE reconstruction error) encourages
    exploration of novel states, helping with the sparse reward problem.
    """

    suffix = "VAE_DQN"   # matches convention in agent_random.py

    def __init__(
        self,
        name,
        this_agent_folder="results/models",
        verbose_console=False,
        verbose_log=True,
        log_directory="results/logs",
        use_sufix=True,
        # ── DQN hyperparameters ──────────────────────────────────────────
        latent_dim=32,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9998,
        batch_size=64,
        buffer_capacity=50_000,
        target_update_freq=500,
        # ── Generative AI (VAE curiosity) ────────────────────────────────
        use_intrinsic_reward=True,
        intrinsic_scale=0.05,
        vae_path=None,
        # ── Mode ─────────────────────────────────────────────────────────
        training=True,
    ):
        super().__init__(
            self.suffix,
            name,
            this_agent_folder,
            verbose_console,
            verbose_log,
            log_directory,
            use_sufix,
        )

        self.latent_dim          = latent_dim
        self.gamma               = gamma
        self.epsilon             = epsilon
        self.epsilon_min         = epsilon_min
        self.epsilon_decay       = epsilon_decay
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_scale     = intrinsic_scale
        self.training            = training
        self.save_dir            = this_agent_folder
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Official reward function (same as agent_random.py) ───────────
        #self.reward_fn = RewardOnlyWinning()
        self.reward_fn = RewardPerformanceScore()

        # ── VAE — Generative AI component ────────────────────────────────
        self.vae = VAE(input_dim=OBS_DIM, hidden_dim=128, latent_dim=latent_dim).to(self.device)
        if vae_path and os.path.exists(vae_path):
            try:
                self.vae.load(vae_path)
            except RuntimeError as e:
                print(f"[WARNING] VAE load failed (size mismatch?): {e}\n"
                      f"[WARNING] Continuing with randomly initialised VAE.")
        self.vae.eval()  # frozen during RL training

        # ── Dueling Double DQN ───────────────────────────────────────────
        self.policy_net = DuelingDQN(latent_dim, ACTION_DIM).to(self.device)
        self.target_net = DuelingDQN(latent_dim, ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        # ── Tracking ─────────────────────────────────────────────────────
        self.total_steps    = 0
        self.losses         = []
        self.match_rewards  = []
        self.episode_rewards = []
        self.win_counts     = {"chef": 0, "sous_chef": 0, "waiter": 0, "dishwasher": 0}

        # Per-step state
        self._last_obs    = None
        self._last_action = None
        self._ep_reward   = 0.0

        self.log(f"VaeDqnAgent ready | device={self.device} | latent={latent_dim} | training={training}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core interface — exact same method signatures as agent_random.py
    # ─────────────────────────────────────────────────────────────────────────

    def get_action(self, observations):
        """
        Called every turn by ChefsHatRoomLocal.
        observations is a 228-dim numpy array (same as agent_random.py receives).

        Process:
          1. Read valid action mask from obs[28:228]
          2. Encode obs → 32-dim latent via VAE
          3. Epsilon-greedy: random valid action OR argmax Q over valid actions
        """
        obs  = np.array(observations, dtype=np.float32)
        mask = obs[28:]   # valid action mask — exactly as in agent_random.py

        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) == 0:
            valid_idx = np.array([ACTION_DIM - 1])   # fallback: pass

        # ── Encode via frozen VAE ─────────────────────────────────────────
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.get_latent(obs_t)   # (1, 32)

        # ── Epsilon-greedy ────────────────────────────────────────────────
        if self.training and np.random.random() < self.epsilon:
            chosen = int(np.random.choice(valid_idx))
        else:
            with torch.no_grad():
                q = self.policy_net(latent).squeeze(0).cpu().numpy()
            q_masked         = np.full(ACTION_DIM, -1e9, dtype=np.float32)
            q_masked[valid_idx] = q[valid_idx]
            chosen           = int(np.argmax(q_masked))

        # Safety: guarantee action is always valid
        if mask[chosen] != 1 and len(valid_idx) > 0:
            chosen = int(np.random.choice(valid_idx))

        self._last_obs    = obs
        self._last_action = chosen

        # Return one-hot array — same format as agent_random.py
        a        = np.zeros(ACTION_DIM, dtype=np.float32)
        a[chosen] = 1
        return a

    def get_reward(self, envInfo):
        """
        Called after each step by ChefsHatRoomLocal.
        Uses the OFFICIAL reward interface (same as agent_random.py).

        envInfo keys used:
            "Author_Index"     → which player index we are
            "Match_Score"      → list of current scores per player
            "Finished_Players" → list of bool, whether each player has finished
        """
        this_player          = envInfo["Author_Index"]
        this_player_position = 3 - envInfo["Match_Score"][this_player]
        this_player_finished = envInfo["Finished_Players"][this_player]

        perf_score = envInfo.get("Game_Performance_Score", [0,0,0,0])
        my_idx = envInfo["Author_Index"]

        return self.reward_fn.getReward(
            this_player_position,
            perf_score[my_idx] if my_idx < len(perf_score) else 0,
            this_player_finished
            )

        #return self.reward_fn.getReward(this_player_position, this_player_finished)


    def get_exhanged_cards(self, cards, amount):
        """
        Return cards to give away at match start (Chef/Sous-Chef role).
        Strategy: give away highest-value cards (same logic as agent_random.py).
        """
        selected = sorted(cards[-amount:])
        self.log(f"Exchanging {amount} cards: {selected}")
        return selected

    def update_start_match(self, cards, players, starting_player):
        """Called at start of each match with initial hand."""
        self._last_obs    = None
        self._last_action = None
        self._ep_reward   = 0.0
        self.log(f"Match started. My cards: {cards}")

    def update_exchange_cards(self, cards_sent, cards_received):
        """Called after card exchange to inform what we got."""
        self.log(f"Exchange: sent={cards_sent}, received={cards_received}")

    def do_special_action(self, info, specialAction):
        """Whether to perform a special action — always accept."""
        return True

    def update_my_action(self, envInfo):
        """
        Called after OUR action is executed.
        Stores transition in replay buffer + runs a training step.

        Intrinsic reward = VAE reconstruction error of next observation.
        This gives a bonus for visiting novel/unusual states.
        """
        if self._last_obs is None or self._last_action is None:
            return

        next_obs_raw = envInfo.get("Observation_After", None)
        if next_obs_raw is None or len(next_obs_raw) == 0:
            return

        next_obs = np.array(next_obs_raw, dtype=np.float32)

        # ── Intrinsic curiosity reward (VAE reconstruction error) ─────────
        step_reward = 0.0
        if self.use_intrinsic_reward:
            nobs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            err    = self.vae.reconstruction_error(nobs_t).item()
            step_reward = self.intrinsic_scale * err

        self._ep_reward += step_reward

        self.buffer.push(self._last_obs, self._last_action, step_reward, next_obs, False)
        self._last_obs = next_obs
        self.total_steps += 1

        if self.training and len(self.buffer) >= self.batch_size:
            loss = self._train_step()
            self.losses.append(loss)

        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.training and self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_action_others(self, envInfo):
        """Called when another player acts — we just observe, no learning."""
        pass

    def update_end_match(self, envInfo):
        """
        Called at end of each match.
        Terminal reward based on finishing position using official reward class.
        """
        if self._last_obs is not None and self._last_action is not None:
            terminal_reward = self.get_reward(envInfo)
            self._ep_reward += terminal_reward

            dummy_next = self._last_obs.copy()
            self.buffer.push(
                self._last_obs, self._last_action,
                terminal_reward, dummy_next, True
            )
            self.match_rewards.append(terminal_reward)

            # Track win counts from Match_Score
            scores = envInfo.get("Match_Score", [])
            if scores:
                my_idx   = envInfo.get("Author_Index", 0)
                my_score = scores[my_idx] if my_idx < len(scores) else -1
                role_map = {3: "chef", 2: "sous_chef", 1: "waiter", 0: "dishwasher"}
                role     = role_map.get(my_score, "dishwasher")
                self.win_counts[role] += 1

        self.episode_rewards.append(self._ep_reward)
        self._ep_reward = 0.0

        if self.training and len(self.buffer) >= self.batch_size:
            self._train_step()

        self.log(f"Match over | epsilon={self.epsilon:.4f} | steps={self.total_steps}")

    def update_game_over(self):
        """Called when entire game ends."""
        if self.training:
            self.save_model()
        self.log(f"Game over | total steps: {self.total_steps}")

    def observe_special_action(self, action_type, player):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Internal training
    # ─────────────────────────────────────────────────────────────────────────

    def _train_step(self):
        """Double DQN update using a sampled minibatch."""
        obs, acts, rews, nobs, dones = self.buffer.sample(self.batch_size)

        obs_t  = torch.FloatTensor(obs).to(self.device)
        nobs_t = torch.FloatTensor(nobs).to(self.device)
        acts_t = torch.LongTensor(acts).to(self.device)
        rews_t = torch.FloatTensor(rews).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            z      = self.vae.get_latent(obs_t)
            nz     = self.vae.get_latent(nobs_t)
            best_a = self.policy_net(nz).argmax(dim=1)
            q_next = self.target_net(nz).gather(1, best_a.unsqueeze(1)).squeeze(1)
            target = rews_t + self.gamma * q_next * (1 - done_t)

        q_curr = self.policy_net(z).gather(1, acts_t.unsqueeze(1)).squeeze(1)
        loss   = torch.nn.functional.smooth_l1_loss(q_curr, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, tag="latest"):
        path = os.path.join(self.save_dir, f"dqn_{tag}.pt")
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim":  self.optimizer.state_dict(),
            "eps":    self.epsilon,
            "steps":  self.total_steps,
        }, path)
        self.log(f"Model saved → {path}")

    def load_model(self, tag="latest"):
        path = os.path.join(self.save_dir, f"dqn_{tag}.pt")
        if not os.path.exists(path):
            self.log(f"No model found at {path}")
            return
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck["policy"])
        self.target_net.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optim"])
        self.epsilon    = ck.get("eps",   self.epsilon_min)
        self.total_steps = ck.get("steps", 0)
        self.log(f"Model loaded ← {path}")

    def get_metrics(self):
        return {
            "episode_rewards": self.episode_rewards,
            "match_rewards":   self.match_rewards,
            "losses":          self.losses,
            "win_counts":      self.win_counts,
            "total_steps":     self.total_steps,
            "final_epsilon":   self.epsilon,
        }
