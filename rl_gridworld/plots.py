# plots.py
from __future__ import annotations
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x: np.ndarray, window: int = 200) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")

def plot_file(npz_path: str, out_dir: str = "figures", ma_window: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    returns = data["returns"]      # [seeds, episodes]
    successes = data["successes"]  # [seeds, episodes]
    trap_hits = data["trap_hits"]  # [seeds, episodes]
    algo = str(data["algo"])
    env_eps = float(data["env_eps"])
    gamma = float(data["gamma"])

    mean_ret = returns.mean(axis=0)
    std_ret = returns.std(axis=0)

    mean_succ = successes.mean(axis=0)
    mean_trap = trap_hits.mean(axis=0)

    ma = moving_average(mean_ret, ma_window)
    ma_x = np.arange(len(ma)) + (ma_window - 1)

    plt.figure()
    plt.plot(ma_x, ma)
    plt.xlabel("Episode")
    plt.ylabel(f"Return (moving avg, w={ma_window})")
    plt.title(f"{algo.upper()} | env ε={env_eps} | γ={gamma}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{algo}_eps{env_eps}_g{gamma}_return.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(mean_succ)
    plt.xlabel("Episode")
    plt.ylabel("Success rate (mean over seeds)")
    plt.title(f"{algo.upper()} | env ε={env_eps} | γ={gamma}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{algo}_eps{env_eps}_g{gamma}_success.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(mean_trap)
    plt.xlabel("Episode")
    plt.ylabel("Trap rate (mean over seeds)")
    plt.title(f"{algo.upper()} | env ε={env_eps} | γ={gamma}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{algo}_eps{env_eps}_g{gamma}_trap.png", dpi=200)
    plt.close()

def main():
    files = glob.glob("results/*.npz")
    if not files:
        raise RuntimeError("No results found. Run experiments.py first.")
    for f in files:
        plot_file(f)
        print("Plotted:", f)

if __name__ == "__main__":
    main()
