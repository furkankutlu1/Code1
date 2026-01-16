# agents.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EpsilonSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995  # multiply each episode

    def value(self, episode: int) -> float:
        return max(self.eps_end, self.eps_start * (self.eps_decay ** episode))

class TabularAgent:
    def __init__(self, n_states: int, n_actions: int = 4, alpha: float = 0.3, gamma: float = 0.95, seed: int = 0):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def act(self, s_idx: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))
        q = self.Q[s_idx]
        max_q = q.max()
        candidates = np.flatnonzero(q == max_q)
        return int(self.rng.choice(candidates))

class QLearningAgent(TabularAgent):
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = r if done else (r + self.gamma * self.Q[s_next].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

class SarsaAgent(TabularAgent):
    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        target = r if done else (r + self.gamma * self.Q[s_next, a_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
