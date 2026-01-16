# env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List
import numpy as np

State = Tuple[int, int]   # (x, y)
Action = int              # 0:up, 1:down, 2:left, 3:right

@dataclass
class StepResult:
    next_state: State
    reward: float
    done: bool
    info: Dict

class StochasticGridworld:
    """
    Episodic stochastic gridworld MDP.

    - Grid size: width x height
    - States: (x, y)
    - Actions: up/down/left/right
    - Stochastic transitions:
        with prob (1 - env_eps): intended action
        with prob env_eps: lateral action (orthogonal), split equally
    - Rewards:
        goal: +1.0 (terminal)
        trap: -1.0 (terminal)
        step: -0.04 otherwise
    """

    ACTIONS = {
        0: (0, -1),  # up
        1: (0,  1),  # down
        2: (-1, 0),  # left
        3: (1,  0),  # right
    }

    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        start: State = (0, 7),
        goal: State = (7, 0),
        traps: Optional[Set[State]] = None,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        trap_penalty: float = -1.0,
        env_eps: float = 0.3,      # transition stochasticity (ε in the MDP)
        seed: int = 42,
    ):
        self.w = int(width)
        self.h = int(height)
        self.start = start
        self.goal = goal

        # Default failure-mode: a "risky corridor" near the top.
        # Short path tends to go through y=1; traps sit just below it at y=2,
        # so stochastic lateral slips can drop agent into traps.
        if traps is None:
            self.traps = set()
            for x in range(1, 7):  # traps at (1..6, 2)
                self.traps.add((x, 2))
            # add a couple more to amplify risk
            self.traps.update({(6, 1), (6, 2)})
        else:
            self.traps = set(traps)

        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.trap_penalty = float(trap_penalty)
        self.env_eps = float(env_eps)

        self.rng = np.random.default_rng(seed)
        self.state: State = self.start

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        return self.state

    def _in_bounds(self, s: State) -> bool:
        x, y = s
        return 0 <= x < self.w and 0 <= y < self.h

    def _move(self, s: State, a: Action) -> State:
        dx, dy = self.ACTIONS[a]
        ns = (s[0] + dx, s[1] + dy)
        if not self._in_bounds(ns):
            return s
        return ns

    def _lateral_actions(self, a: Action) -> List[Action]:
        # Lateral actions are orthogonal to intended action
        if a in (0, 1):  # up/down -> left/right
            return [2, 3]
        return [0, 1]    # left/right -> up/down

    def step(self, a_intended: Action) -> StepResult:
        s = self.state

        p = self.rng.random()
        if p < (1.0 - self.env_eps):
            a_chosen = a_intended
        else:
            lat = self._lateral_actions(a_intended)
            a_chosen = lat[0] if self.rng.random() < 0.5 else lat[1]

        s_next = self._move(s, a_chosen)

        done = False
        if s_next == self.goal:
            r = self.goal_reward
            done = True
        elif s_next in self.traps:
            r = self.trap_penalty
            done = True
        else:
            r = self.step_cost

        self.state = s_next
        return StepResult(
            next_state=s_next,
            reward=r,
            done=done,
            info={"chosen_action": a_chosen}
        )

    def state_to_idx(self, s: State) -> int:
        x, y = s
        return y * self.w + x

    def idx_to_state(self, idx: int) -> State:
        x = idx % self.w
        y = idx // self.w
        return (x, y)

    @property
    def n_states(self) -> int:
        return self.w * self.h

    @property
    def n_actions(self) -> int:
        return 4
