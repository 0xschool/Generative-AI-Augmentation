"""
Dueling Double DQN network operating on VAE latent codes (32-dim → 200 Q-values).

Dueling architecture separates state-value V(s) from action advantages A(s,a),
which helps in Chef's Hat where many actions have similar value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    def __init__(self, input_dim=32, action_dim=200, hidden_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, z):
        f   = self.features(z)
        v   = self.value_head(f)
        a   = self.advantage_head(f)
        # Q(s,a) = V(s) + A(s,a) − mean(A(s,:))
        return v + a - a.mean(dim=1, keepdim=True)
