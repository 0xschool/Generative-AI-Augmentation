"""
VAE-Augmented DQN Agent — Variant 6: Generative AI Augmentation
Student ID: 16735347

Follows exact same interface as official ChefsHatGYM agent_random.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
# Reward computed directly — RewardOnlyWinning gives same value for positions 1-3

from models.vae import VAE
from models.dqn import DuelingDQN
from models.replay_buffer import ReplayBuffer

OBS_DIM    = 228
ACTION_DIM = 200


class VaeDqnAgent(ChefsHatPlayer):

    suffix = "VAE_DQN"

    def __init__(
        self,
        name,
        this_agent_folder="results/models",
        verbose_console=False,
        verbose_log=True,
        log_directory="results/logs",
        use_sufix=True,
        latent_dim=32,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        batch_size=64,
        buffer_capacity=100_000,
        target_update_freq=100,
        use_intrinsic_reward=True,
        intrinsic_scale=0.05,
        vae_path=None,
        training=True,
    ):
        super().__init__(
            self.suffix, name, this_agent_folder,
            verbose_console, verbose_log, log_directory, use_sufix,
        )

        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_scale    = intrinsic_scale
        self.training           = training
        self.save_dir           = this_agent_folder
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Graded terminal rewards — distinct signal for every finishing position

        # VAE — Generative AI component
        self.vae = VAE(OBS_DIM, 128, latent_dim).to(self.device)
        if vae_path and os.path.exists(vae_path):
            try:
                self.vae.load(vae_path)
            except RuntimeError as e:
                print(f"[VAE] Load failed: {e}. Using random init.")
        self.vae.eval()

        # Dueling Double DQN
        self.policy_net = DuelingDQN(latent_dim, ACTION_DIM).to(self.device)
        self.target_net = DuelingDQN(latent_dim, ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        # Tracking
        self.total_steps    = 0
        self.match_count    = 0
        self.losses         = []
        self.match_rewards  = []
        self.win_counts     = {"chef": 0, "sous_chef": 0, "waiter": 0, "dishwasher": 0}

        # Per-step state — set in get_action, used in update_my_action
        self._last_obs      = None
        self._last_action   = None
        self._last_hand     = None
        self._match_reward  = 0.0

    # ── Official interface (same as agent_random.py) ──────────────────────────

    def get_action(self, observations):
        """Called every turn. observations = 228-dim array."""
        obs  = np.array(observations, dtype=np.float32)
        mask = obs[28:]

        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) == 0:
            valid_idx = np.array([ACTION_DIM - 1])

        # Encode via frozen VAE
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.get_latent(obs_t)

        # Epsilon-greedy
        if self.training and np.random.random() < self.epsilon:
            chosen = int(np.random.choice(valid_idx))
        else:
            with torch.no_grad():
                q = self.policy_net(latent).squeeze(0).cpu().numpy()
            q_m            = np.full(ACTION_DIM, -1e9, dtype=np.float32)
            q_m[valid_idx] = q[valid_idx]
            chosen         = int(np.argmax(q_m))
            if mask[chosen] != 1:
                chosen = int(np.random.choice(valid_idx))

        # Save for update_my_action
        self._last_obs    = obs
        self._last_action = chosen
        self._last_hand   = int(np.sum(obs[11:28] > 0))

        a         = np.zeros(ACTION_DIM, dtype=np.float32)
        a[chosen] = 1
        return a

    def get_reward(self, envInfo):
        """
        Graded terminal reward — distinct value for each finishing position.
        RewardOnlyWinning returns -0.001 for positions 1,2,3 (no gradient),
        so we compute our own graded signal instead.
        """
        my_idx  = envInfo.get("Author_Index", 0)
        scores  = envInfo.get("Match_Score", [0]*4)
        score   = scores[my_idx] if my_idx < len(scores) else 0
        # score: 3=chef, 2=sous_chef, 1=waiter, 0=dishwasher
        terminal_map = {3: 1.0, 2: 0.5, 1: -0.5, 0: -1.0}
        return terminal_map.get(int(score), -1.0)

    def get_exhanged_cards(self, cards, amount):
        selected = sorted(cards[-amount:])
        return selected

    def update_start_match(self, cards, players, starting_player):
        self._last_obs     = None
        self._last_action  = None
        self._last_hand    = None
        self._match_reward = 0.0

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        return True

    def update_my_action(self, envInfo):
        """
        Called after OUR action is executed by the room.
        The room provides Observation_Before and Observation_After in envInfo,
        so we use the real next_obs for proper Q-learning transitions.
        """
        if self._last_obs is None or self._last_action is None:
            return

        my_idx = envInfo.get("Author_Index", 0)

        # Get real next observation from room (key fix)
        next_obs_raw = envInfo.get("Observation_After", None)
        if next_obs_raw is not None:
            next_obs = np.array(next_obs_raw, dtype=np.float32)
        else:
            next_obs = self._last_obs   # fallback

        # Shaped step reward
        step_reward = -0.002

        # Card progress reward
        cards_now = envInfo.get("Cards_Per_Player", None)
        if cards_now and self._last_hand is not None:
            hand_now = cards_now[my_idx] if my_idx < len(cards_now) else self._last_hand
            if hand_now < self._last_hand:
                step_reward += 0.01 * (self._last_hand - hand_now)
            self._last_hand = hand_now

        # Pizza bonus
        if envInfo.get("Is_Pizza", False) and envInfo.get("Pizza_Author", -1) == my_idx:
            step_reward += 0.05

        # VAE intrinsic curiosity reward
        if self.use_intrinsic_reward:
            obs_t = torch.FloatTensor(self._last_obs).unsqueeze(0).to(self.device)
            err   = self.vae.reconstruction_error(obs_t).item()
            step_reward += self.intrinsic_scale * err

        self._match_reward += step_reward

        # Store transition with REAL next_obs
        self.buffer.push(self._last_obs, self._last_action, step_reward,
                         next_obs, 0.0)
        self._last_obs = next_obs
        self.total_steps += 1

        if self.training and len(self.buffer) >= self.batch_size:
            loss = self._train_step()
            if loss:
                self.losses.append(loss)

        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.training and self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_action_others(self, envInfo):
        pass

    def update_end_match(self, envInfo):
        """
        Called at match end. This is where the REAL learning signal comes from.
        Terminal reward based on finishing position.
        """
        if self._last_obs is None or self._last_action is None:
            return

        # Terminal reward from official reward function
        terminal_reward = self.get_reward(envInfo)
        self._match_reward += terminal_reward

        # Store terminal transition with done=True
        self.buffer.push(self._last_obs, self._last_action, terminal_reward,
                         self._last_obs, 1.0)

        self.match_rewards.append(self._match_reward)
        self.match_count += 1

        # Record finishing position
        my_idx   = envInfo.get("Author_Index", 0)
        scores   = envInfo.get("Match_Score", [0]*4)
        pos      = 3 - (scores[my_idx] if my_idx < len(scores) else 0)
        role_map = {0:"chef", 1:"sous_chef", 2:"waiter", 3:"dishwasher"}
        self.win_counts[role_map.get(pos, "dishwasher")] += 1

        # Extra training at match end
        if self.training:
            for _ in range(4):
                if len(self.buffer) >= self.batch_size:
                    loss = self._train_step()
                    if loss:
                        self.losses.append(loss)

        # Print progress every 100 matches
        if self.match_count % 100 == 0:
            wc  = self.win_counts
            tot = sum(wc.values())
            avg = float(np.mean(self.match_rewards[-100:])) if self.match_rewards else 0
            print(f"  [Match {self.match_count}] "
                  f"Chef={wc['chef']}({100*wc['chef']/max(tot,1):.0f}%) | "
                  f"Dish={wc['dishwasher']}({100*wc['dishwasher']/max(tot,1):.0f}%) | "
                  f"AvgR={avg:.3f} | ε={self.epsilon:.3f} | "
                  f"Buf={len(self.buffer):,}")

        self._match_reward = 0.0

    def update_game_over(self):
        if self.training:
            self.save_model()

    def observe_special_action(self, action_type, player):
        pass

    # ── DQN update ────────────────────────────────────────────────────────────

    def _train_step(self):
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
        loss   = F.smooth_l1_loss(q_curr, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_model(self, tag="latest"):
        path = os.path.join(self.save_dir, f"dqn_{tag}.pt")
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim":  self.optimizer.state_dict(),
            "eps":    self.epsilon,
            "steps":  self.total_steps,
        }, path)
        print(f"  [Saved] {path}")

    def load_model(self, tag="latest"):
        path = os.path.join(self.save_dir, f"dqn_{tag}.pt")
        if not os.path.exists(path):
            return
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck["policy"])
        self.target_net.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optim"])
        self.epsilon     = ck.get("eps",   self.epsilon_min)
        self.total_steps = ck.get("steps", 0)
        print(f"  [Loaded] {path}")

    def get_metrics(self):
        return {
            "match_rewards":  self.match_rewards,
            "losses":         self.losses,
            "win_counts":     self.win_counts,
            "total_steps":    self.total_steps,
            "final_epsilon":  self.epsilon,
        }
