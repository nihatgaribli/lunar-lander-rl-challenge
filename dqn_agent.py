"""
DQN Agent for LunarLander-v2 (Track V).

Implements Deep Q-Network with:
  - Experience Replay Buffer
  - Target Network (soft update)
  - Epsilon-greedy exploration with linear decay

All code is written from scratch — no RL libraries used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    """
    Q-value approximator: MLP that maps 8-D observations to Q-values
    for each of the 4 discrete actions.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.
    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with experience replay and target network.

    Hyperparameters tuned for LunarLander-v2 to reliably solve
    (avg return >= 200) within ~150K–250K environment steps.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Step counter
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        When evaluate=True, always picks greedy action (no exploration).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax(dim=1).item()

    def _update_epsilon(self):
        """Linear epsilon decay."""
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def step(self, state, action, reward, next_state, done):
        """
        Store transition and perform one training step if conditions are met.
        Returns the loss value (or None if no update was performed).
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._update_epsilon()

        loss = None
        if (
            len(self.replay_buffer) >= self.learning_starts
            and self.total_steps % self.train_freq == 0
        ):
            loss = self._train_step()

        return loss

    def _train_step(self) -> float:
        """Sample a batch from replay buffer and perform one gradient step."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values: Q(s, a) for the actions actually taken
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_max = self.target_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # MSE loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        return loss.item()

    def _soft_update(self):
        """Polyak averaging: θ_target ← τ·θ + (1−τ)·θ_target"""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]
