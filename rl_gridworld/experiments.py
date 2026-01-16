# experiments.py
from __future__ import annotations
import os
import numpy as np

from env import StochasticGridworld
from agents import EpsilonSchedule
from train import train_q_learning, train_sarsa

def run_one(
    algo: str,
    env_eps: float,
    gamma: float,
    seed: int,
    episodes: int = 6000,
    alpha: float = 0.3,
    max_steps: int = 400,
):
    env = StochasticGridworld(env_eps=env_eps, seed=seed)
    eps_sched = EpsilonSchedule(eps_start=1.0, eps_end=0.05, eps_decay=0.995)

    if algo.lower() == "q":
        agent, metrics = train_q_learning(env, episodes=episodes, alpha=alpha, gamma=gamma,
                                          eps_schedule=eps_sched, seed=seed, max_steps=max_steps)
    elif algo.lower() == "sarsa":
        agent, metrics = train_sarsa(env, episodes=episodes, alpha=alpha, gamma=gamma,
                                     eps_schedule=eps_sched, seed=seed, max_steps=max_steps)
    else:
        raise ValueError("algo must be 'q' or 'sarsa'")

    return metrics, agent.Q

def main():
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    env_eps_list = [0.0, 0.1, 0.3, 0.5]
    gamma_list = [0.8, 0.95, 0.99]
    seeds = [0, 1, 2, 3, 4]  # 5 seeds

    episodes = 6000
    alpha = 0.3
    max_steps = 400

    for env_eps in env_eps_list:
        for gamma in gamma_list:
            for algo in ["q", "sarsa"]:
                all_returns = []
                all_success = []
                all_trap = []
                all_lengths = []

                for seed in seeds:
                    metrics, Q = run_one(algo, env_eps, gamma, seed, episodes=episodes, alpha=alpha, max_steps=max_steps)
                    all_returns.append(metrics["returns"])
                    all_success.append(metrics["successes"])
                    all_trap.append(metrics["trap_hits"])
                    all_lengths.append(metrics["lengths"])

                all_returns = np.stack(all_returns)  # [seeds, episodes]
                all_success = np.stack(all_success)
                all_trap = np.stack(all_trap)
                all_lengths = np.stack(all_lengths)

                fname = f"{out_dir}/{algo}_eps{env_eps}_g{gamma}.npz"
                np.savez_compressed(
                    fname,
                    returns=all_returns,
                    successes=all_success,
                    trap_hits=all_trap,
                    lengths=all_lengths,
                    env_eps=env_eps,
                    gamma=gamma,
                    algo=algo,
                    episodes=episodes,
                    alpha=alpha,
                    max_steps=max_steps,
                    seeds=np.array(seeds),
                )
                print("Saved:", fname)

if __name__ == "__main__":
    main()
