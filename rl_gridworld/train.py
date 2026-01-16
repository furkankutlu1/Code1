# train.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from env import StochasticGridworld
from agents import QLearningAgent, SarsaAgent, EpsilonSchedule

def train_q_learning(
    env: StochasticGridworld,
    episodes: int = 5000,
    alpha: float = 0.3,
    gamma: float = 0.95,
    eps_schedule: Optional[EpsilonSchedule] = None,
    seed: int = 0,
    max_steps: int = 500,
) -> Tuple[QLearningAgent, Dict[str, np.ndarray]]:
    if eps_schedule is None:
        eps_schedule = EpsilonSchedule()

    agent = QLearningAgent(env.n_states, env.n_actions, alpha=alpha, gamma=gamma, seed=seed)
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []
    trap_hits: List[int] = []

    # separate env seed stream from agent seed stream
    env.reset(seed=seed + 10_000)

    for ep in range(episodes):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        eps = eps_schedule.value(ep)

        G = 0.0
        trap = 0
        success = 0

        for t in range(max_steps):
            a = agent.act(s_idx, eps)
            res = env.step(a)
            s_next_idx = env.state_to_idx(res.next_state)

            agent.update(s_idx, a, res.reward, s_next_idx, res.done)

            G += res.reward
            s_idx = s_next_idx

            if res.done:
                if res.next_state == env.goal:
                    success = 1
                elif res.next_state in env.traps:
                    trap = 1
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)

        returns.append(G)
        successes.append(success)
        trap_hits.append(trap)

    metrics = {
        "returns": np.array(returns, dtype=np.float64),
        "lengths": np.array(lengths, dtype=np.int32),
        "successes": np.array(successes, dtype=np.int32),
        "trap_hits": np.array(trap_hits, dtype=np.int32),
    }
    return agent, metrics

def train_sarsa(
    env: StochasticGridworld,
    episodes: int = 5000,
    alpha: float = 0.3,
    gamma: float = 0.95,
    eps_schedule: Optional[EpsilonSchedule] = None,
    seed: int = 0,
    max_steps: int = 500,
) -> Tuple[SarsaAgent, Dict[str, np.ndarray]]:
    if eps_schedule is None:
        eps_schedule = EpsilonSchedule()

    agent = SarsaAgent(env.n_states, env.n_actions, alpha=alpha, gamma=gamma, seed=seed)
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []
    trap_hits: List[int] = []

    env.reset(seed=seed + 10_000)

    for ep in range(episodes):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        eps = eps_schedule.value(ep)

        a = agent.act(s_idx, eps)
        G = 0.0
        trap = 0
        success = 0

        for t in range(max_steps):
            res = env.step(a)
            s_next_idx = env.state_to_idx(res.next_state)

            if res.done:
                agent.update(s_idx, a, res.reward, s_next_idx, a_next=0, done=True)
                G += res.reward
                if res.next_state == env.goal:
                    success = 1
                elif res.next_state in env.traps:
                    trap = 1
                lengths.append(t + 1)
                break

            a_next = agent.act(s_next_idx, eps)
            agent.update(s_idx, a, res.reward, s_next_idx, a_next=a_next, done=False)

            G += res.reward
            s_idx = s_next_idx
            a = a_next
        else:
            lengths.append(max_steps)

        returns.append(G)
        successes.append(success)
        trap_hits.append(trap)

    metrics = {
        "returns": np.array(returns, dtype=np.float64),
        "lengths": np.array(lengths, dtype=np.int32),
        "successes": np.array(successes, dtype=np.int32),
        "trap_hits": np.array(trap_hits, dtype=np.int32),
    }
    return agent, metrics
