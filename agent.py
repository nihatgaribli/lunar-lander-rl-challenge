"""
Submission interface for LunarLander-v2 grading harness.

Exposes:
  - load_policy(checkpoint_path: str) -> Policy
  - Policy.select_action(observation: np.ndarray) -> int

The grading harness imports this file, calls load_policy once,
then calls select_action once per environment step across all
100 private-seed episodes.
"""

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-value network (must match training architecture exactly)."""

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


class Policy:
    """
    Wraps a trained Q-network for greedy action selection.
    """

    def __init__(self, q_network: QNetwork, device: torch.device):
        self.q_network = q_network
        self.device = device
        self.q_network.eval()

    def select_action(self, observation: np.ndarray) -> int:
        """
        Greedy/deterministic action selection.
        Returns one of the 4 discrete actions: {0, 1, 2, 3}.
        No exploration noise at evaluation time.
        """
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax(dim=1).item()


def load_policy(checkpoint_path: str) -> Policy:
    """
    Load trained DQN weights from checkpoint and return a Policy object.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        Policy object with select_action method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create network with same architecture as training
    q_network = QNetwork(state_dim=8, action_dim=4, hidden=128).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    q_network.load_state_dict(checkpoint["q_network_state_dict"])

    return Policy(q_network, device)
