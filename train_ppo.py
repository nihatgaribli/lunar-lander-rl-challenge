"""
PPO Training Script for LunarLander-v3 (Bonus Track P).

Trains PPO agent with multiple seeds, evaluates on public seeds,
saves checkpoints, and generates learning curve plots.
"""

import os
import json
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ppo_agent import PPOAgent

# ─────────────────────────── Configuration ───────────────────────────

PUBLIC_SEEDS = [
    7404, 13783, 19113, 21818, 33091, 42191, 46784, 46841, 48180, 52582,
    54637, 55510, 62379, 65851, 79439, 80265, 82779, 87379, 94258, 95232,
]

TRAINING_SEEDS = [42, 123, 456]

# PPO gets a separate budget (bonus track).
# We allocate 150K steps per seed.
MAX_STEPS_PER_SEED = 150_000

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# PPO Hyperparameters
PPO_CONFIG = {
    "state_dim": 8,
    "action_dim": 4,
    "hidden": 64,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 2048,
    "n_epochs": 4,
    "minibatch_size": 64,
    "anneal_lr": True,
    "total_timesteps": 150_000,
}


def evaluate_policy(agent, seeds):
    """Evaluate agent on given seeds with greedy policy."""
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


def train_one_seed(seed: int, max_steps: int):
    """Train PPO agent with a single seed."""
    print(f"\n{'='*60}")
    print(f"Training PPO with seed {seed} (max {max_steps} steps)")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = PPOAgent(**PPO_CONFIG)
    env = gym.make("LunarLander-v3")

    episode_returns = []
    step_checkpoints = []
    steps_used = 0
    episode = 0

    state, _ = env.reset(seed=seed)
    current_episode_return = 0.0

    while steps_used < max_steps:
        # Collect rollout of n_steps
        for _ in range(agent.n_steps):
            if steps_used >= max_steps:
                break

            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.store(state, action, log_prob, reward, float(done), value)
            agent.total_steps += 1
            steps_used += 1

            current_episode_return += reward

            if done:
                episode_returns.append(current_episode_return)
                episode += 1
                current_episode_return = 0.0
                state, _ = env.reset(seed=seed + episode)

                if episode % 50 == 0 and len(episode_returns) >= 50:
                    recent_avg = np.mean(episode_returns[-50:])
                    print(
                        f"  Episode {episode:4d} | Steps: {steps_used:>7d} | "
                        f"Avg Return (50): {recent_avg:7.1f}"
                    )
                    step_checkpoints.append((steps_used, recent_avg))
            else:
                state = next_state

        # Compute last value for GAE bootstrap
        if len(agent.buffer) > 0:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                last_value = agent.policy.get_value(state_t).item()

            # Track if current state is terminal
            last_done = 0.0  # we are mid-episode

            # PPO update
            losses = agent.update(last_value, last_done)

    env.close()

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_seed_{seed}.pt")
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
    """Plot learning curves across all PPO seeds."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    for result in all_results:
        returns = result["episode_returns"]
        window = 50
        if len(returns) >= window:
            rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax1.plot(rolling, label=f"Seed {result['seed']}", alpha=0.8)
        else:
            ax1.plot(returns, label=f"Seed {result['seed']}", alpha=0.8)

    ax1.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Solved (200)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return (50-episode rolling avg)")
    ax1.set_title("PPO Learning Curves (per seed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    seeds = [r["seed"] for r in all_results]
    means = [r["public_eval_mean"] for r in all_results]
    stds = [r["public_eval_std"] for r in all_results]
    colors = ["#9C27B0", "#E91E63", "#00BCD4"]

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
    ax2.set_title("PPO Public Seed Evaluation")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

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
    plot_path = os.path.join(PLOTS_DIR, "ppo_learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {plot_path}")


def main():
    print("=" * 60)
    print("  PPO Training for LunarLander-v3 (Track P — Bonus)")
    print(f"  Training seeds: {TRAINING_SEEDS}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    all_results = []
    best_agent = None
    best_mean = -float("inf")
    best_seed = None

    for seed in TRAINING_SEEDS:
        result, agent = train_one_seed(seed, MAX_STEPS_PER_SEED)
        all_results.append(result)

        if result["public_eval_mean"] > best_mean:
            best_mean = result["public_eval_mean"]
            best_agent = agent
            best_seed = seed

    # Save best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "ppo_best.pt")
    best_agent.save(best_checkpoint)
    total_steps = sum(r["steps_used"] for r in all_results)
    print(f"\n{'='*60}")
    print(f"Best PPO agent: seed {best_seed} (mean return: {best_mean:.1f})")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Total PPO steps: {total_steps:,}")
    print(f"{'='*60}")

    plot_learning_curves(all_results)

    # Summary
    all_means = [r["public_eval_mean"] for r in all_results]
    print(f"\nPPO Seed Variance Analysis:")
    print(f"  Mean across seeds: {np.mean(all_means):.1f}")
    print(f"  Std across seeds:  {np.std(all_means):.1f}")
    print(f"  Individual seeds:  {[f'{m:.1f}' for m in all_means]}")

    # Save results
    results_path = os.path.join(CHECKPOINT_DIR, "ppo_results.json")
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
