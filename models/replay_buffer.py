"""Experience replay buffer for DQN training."""

import numpy as np
import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch               = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            np.array(obs,   dtype=np.float32),
            np.array(acts,  dtype=np.int64),
            np.array(rews,  dtype=np.float32),
            np.array(nobs,  dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)
