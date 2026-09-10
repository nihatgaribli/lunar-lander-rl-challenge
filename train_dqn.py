"""
DQN Training Script for LunarLander-v3.

Trains DQN agent with multiple seeds, evaluates on public seeds,
saves checkpoints, and generates learning curve plots.

Complies with:
  - 500,000 total environment step budget
  - 3+ training seeds for seed-variance analysis
  - Evaluation on 20 public seeds
"""

import os
import json
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dqn_agent import DQNAgent

# ─────────────────────────── Configuration ───────────────────────────

# Public evaluation seeds (from project spec)
PUBLIC_SEEDS = [
    7404, 13783, 19113, 21818, 33091, 42191, 46784, 46841, 48180, 52582,
    54637, 55510, 62379, 65851, 79439, 80265, 82779, 87379, 94258, 95232,
]

# Training seeds for seed-variance analysis
TRAINING_SEEDS = [42, 123, 456]

# Budget: 500K total steps across all runs
# With 3 seeds: ~150K steps per seed (conservative to stay in budget)
MAX_STEPS_PER_SEED = 150_000
TOTAL_STEP_BUDGET = 500_000

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# DQN Hyperparameters
DQN_CONFIG = {
    "state_dim": 8,
    "action_dim": 4,
    "hidden": 128,
    "lr": 5e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 80_000,
    "buffer_capacity": 100_000,
    "batch_size": 64,
    "learning_starts": 1_000,
    "train_freq": 1,
}


def evaluate_policy(agent, seeds, render=False):
    """
    Evaluate agent on given seeds (one episode per seed, greedy policy).
    Returns list of episode returns.
    """
    returns = []
    for seed in seeds:
        env = gym.make("LunarLander-v3")
        state, _ = env.reset(seed=seed)
        episode_return = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state
            done = terminated or truncated

        returns.append(episode_return)
        env.close()

    return returns


def sanity_check():
    """
    DQN sanity check from Section 6 of the project spec.
    Verify that the target computation produces y = 16.88.
    """
    r = 5.0
    gamma = 0.99
    max_q_target_next = 12.0

    y = r + gamma * max_q_target_next
    expected = 16.88

    print(f"Sanity Check: y = {r} + {gamma} * {max_q_target_next} = {y}")
    assert abs(y - expected) < 1e-6, f"FAILED: expected {expected}, got {y}"
    print("[OK] Sanity check PASSED (y = 16.88)\n")


def train_one_seed(seed: int, max_steps: int, cumulative_steps: int):
    """
    Train DQN agent with a single seed.
    Returns training history and the trained agent.
    """
    print(f"\n{'='*60}")
    print(f"Training DQN with seed {seed} (max {max_steps} steps)")
    print(f"Cumulative steps so far: {cumulative_steps}")
    print(f"{'='*60}\n")

    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DQNAgent(**DQN_CONFIG)

    env = gym.make("LunarLander-v3")

    # Training history
    episode_returns = []
    episode_steps_list = []
    step_checkpoints = []  # (step, avg_return) for plotting

    episode = 0
    steps_used = 0

    while steps_used < max_steps:
        state, _ = env.reset(seed=seed + episode)  # Vary seed per episode
        episode_return = 0.0
        episode_steps = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, float(done))

            state = next_state
            episode_return += reward
            episode_steps += 1
            steps_used += 1

            if steps_used >= max_steps:
                break

        episode_returns.append(episode_return)
        episode_steps_list.append(episode_steps)
        episode += 1

        # Log progress every 50 episodes
        if episode % 50 == 0:
            recent_avg = np.mean(episode_returns[-50:])
            print(
                f"  Episode {episode:4d} | Steps: {steps_used:>7d} | "
                f"Avg Return (50): {recent_avg:7.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            step_checkpoints.append((steps_used, recent_avg))

    env.close()

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dqn_seed_{seed}.pt")
    agent.save(checkpoint_path)
    print(f"\n  Checkpoint saved: {checkpoint_path}")
    print(f"  Total steps used this seed: {steps_used}")

    # Evaluate on public seeds
    public_returns = evaluate_policy(agent, PUBLIC_SEEDS)
    mean_return = np.mean(public_returns)
    std_return = np.std(public_returns)
    print(f"  Public seed evaluation: {mean_return:.1f} ± {std_return:.1f}")

    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "step_checkpoints": step_checkpoints,
        "steps_used": steps_used,
        "public_eval_mean": mean_return,
        "public_eval_std": std_return,
        "public_eval_returns": public_returns,
        "checkpoint_path": checkpoint_path,
    }, agent


def plot_learning_curves(all_results):
    """Plot learning curves across all seeds with mean ± std shading."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Plot 1: Individual learning curves ──
    ax1 = axes[0]
    for result in all_results:
        returns = result["episode_returns"]
        # Compute rolling average
        window = 50
        if len(returns) >= window:
            rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax1.plot(rolling, label=f"Seed {result['seed']}", alpha=0.8)
        else:
            ax1.plot(returns, label=f"Seed {result['seed']}", alpha=0.8)

    ax1.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Solved (200)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return (50-episode rolling avg)")
    ax1.set_title("DQN Learning Curves (per seed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Public seed evaluation bar chart ──
    ax2 = axes[1]
    seeds = [r["seed"] for r in all_results]
    means = [r["public_eval_mean"] for r in all_results]
    stds = [r["public_eval_std"] for r in all_results]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars = ax2.bar(
        range(len(seeds)),
        means,
        yerr=stds,
        color=colors[: len(seeds)],
        capsize=5,
        alpha=0.8,
    )
    ax2.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Solved (200)")
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel("Mean Return (Public Seeds)")
    ax2.set_title("DQN Public Seed Evaluation")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 5,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "dqn_learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {plot_path}")


def main():
    print("=" * 60)
    print("  DQN Training for LunarLander-v3 (Track V)")
    print(f"  Total step budget: {TOTAL_STEP_BUDGET:,}")
    print(f"  Training seeds: {TRAINING_SEEDS}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # Step 1: Sanity check
    sanity_check()

    # Step 2: Train with multiple seeds
    all_results = []
    cumulative_steps = 0
    best_agent = None
    best_mean = -float("inf")

    for seed in TRAINING_SEEDS:
        remaining_budget = TOTAL_STEP_BUDGET - cumulative_steps
        steps_for_this_seed = min(MAX_STEPS_PER_SEED, remaining_budget)

        if steps_for_this_seed <= 0:
            print(f"\nBudget exhausted! Skipping seed {seed}")
            break

        result, agent = train_one_seed(seed, steps_for_this_seed, cumulative_steps)
        cumulative_steps += result["steps_used"]
        all_results.append(result)

        # Track best
        if result["public_eval_mean"] > best_mean:
            best_mean = result["public_eval_mean"]
            best_agent = agent
            best_seed = seed

    # Step 3: Save best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "dqn_best.pt")
    best_agent.save(best_checkpoint)
    print(f"\n{'='*60}")
    print(f"Best DQN agent: seed {best_seed} (mean return: {best_mean:.1f})")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Total cumulative steps: {cumulative_steps:,} / {TOTAL_STEP_BUDGET:,}")
    print(f"{'='*60}")

    # Step 4: Plot learning curves
    plot_learning_curves(all_results)

    # Step 5: Summary statistics
    all_means = [r["public_eval_mean"] for r in all_results]
    print(f"\nSeed Variance Analysis:")
    print(f"  Mean across seeds: {np.mean(all_means):.1f}")
    print(f"  Std across seeds:  {np.std(all_means):.1f}")
    print(f"  Individual seeds:  {[f'{m:.1f}' for m in all_means]}")

    # Save results as JSON for later use
    results_path = os.path.join(CHECKPOINT_DIR, "dqn_results.json")
    serializable = []
    for r in all_results:
        serializable.append({
            "seed": r["seed"],
            "steps_used": r["steps_used"],
            "public_eval_mean": r["public_eval_mean"],
            "public_eval_std": r["public_eval_std"],
            "public_eval_returns": r["public_eval_returns"],
            "episode_returns": r["episode_returns"],
        })
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {results_path}")

    return all_results


if __name__ == "__main__":
    main()
