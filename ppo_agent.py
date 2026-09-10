"""
PPO Agent for LunarLander-v3 (Track P - Bonus).

Implements Proximal Policy Optimization with:
  - Separate Actor and Critic networks (no weight sharing)
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Orthogonal weight initialization
  - Learning rate annealing

All code is written from scratch - no RL libraries used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (as in CleanRL/PPO best practices)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic with SEPARATE networks for policy and value.
    Uses orthogonal initialization and Tanh activations (standard for PPO).
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden: int = 64):
        super().__init__()

        # Separate actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )

        # Separate critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor, action=None):
        """
        Given a state, return action, log_prob, entropy, and value.
        If action is provided, compute log_prob and entropy for that action.
        """
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s) only."""
        return self.critic(state).squeeze(-1)


class RolloutBuffer:
    """
    Fixed-size buffer for PPO rollout storage.
    Pre-allocates arrays for efficiency.
    """

    def __init__(self, n_steps, state_dim, device):
        self.n_steps = n_steps
        self.device = device
        self.pos = 0

        self.states = np.zeros((n_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)

    def store(self, state, action, log_prob, reward, done, value):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.pos += 1

    def clear(self):
        self.pos = 0

    def get(self):
        """Convert to tensors."""
        n = self.pos
        return (
            torch.FloatTensor(self.states[:n]).to(self.device),
            torch.LongTensor(self.actions[:n]).to(self.device),
            torch.FloatTensor(self.log_probs[:n]).to(self.device),
            self.rewards[:n].copy(),
            self.dones[:n].copy(),
            self.values[:n].copy(),
        )

    def __len__(self):
        return self.pos


class PPOAgent:
    """
    PPO agent with clipped surrogate objective and GAE.
    Hyperparameters tuned for LunarLander-v3.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden: int = 64,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 4,
        minibatch_size: int = 64,
        anneal_lr: bool = True,
        total_timesteps: int = 150_000,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.anneal_lr = anneal_lr
        self.total_timesteps = total_timesteps
        self.initial_lr = lr

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, state_dim, self.device)

        # Step counter
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> tuple:
        """Select action using current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                logits, _ = self.policy(state_t)
                action = logits.argmax(dim=1).item()
                return action
            else:
                action, log_prob, _, value = self.policy.get_action_and_value(state_t)
                return (
                    action.item(),
                    log_prob.item(),
                    value.item(),
                )

    def _anneal_learning_rate(self):
        """Linear LR annealing."""
        if not self.anneal_lr:
            return
        frac = 1.0 - (self.total_steps / self.total_timesteps)
        frac = max(frac, 0.0)
        lr_now = frac * self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now

    def compute_gae(self, rewards, dones, values, last_value, last_done):
        """Compute GAE with proper episode boundary handling."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def update(self, last_value: float, last_done: float):
        """Perform PPO update."""
        self._anneal_learning_rate()

        states, actions, old_log_probs, rewards, dones, values = self.buffer.get()

        # Compute GAE
        advantages, returns = self.compute_gae(
            rewards, dones, values, last_value, last_done
        )

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update: multiple epochs over minibatches
        n = len(states)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                # Get current policy's evaluation
                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(mb_states, mb_actions)
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximate KL for diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.entropy_coef * entropy_loss + self.value_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "v_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
