"""
network.py

Deep Q-Network used by SOR-DQN.

Supports:
- Fully-connected observations
- Easy extension to CNN encoder later
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Standard Deep Q Network.

    Input:
        state : (batch_size, state_dim)

    Output:
        Q-values for every action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        h1, h2 = hidden_dims

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization.
        """

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):

        x = F.relu(self.fc1(state))

        x = F.relu(self.fc2(x))

        q = self.fc3(x)

        return q

    def act(self, state):
        """
        Returns greedy action.

        state:
            torch.Tensor(state_dim)
            OR
            torch.Tensor(batch,state_dim)
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():

            q = self.forward(state)

            action = q.argmax(dim=1)

        return action.item()


class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture.

    Useful if you later want
    SOR-Dueling DQN.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        h1, h2 = hidden_dims

        self.feature = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(h2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(h2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                nn.init.zeros_(m.bias)

    def forward(self, state):

        features = self.feature(state)

        value = self.value_stream(features)

        advantage = self.advantage_stream(features)

        q = value + advantage - advantage.mean(
            dim=1,
            keepdim=True,
        )

        return q

    def act(self, state):

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():

            q = self.forward(state)

            action = q.argmax(dim=1)

        return action.item()


if __name__ == "__main__":

    state_dim = 64
    action_dim = 5

    model = QNetwork(
        state_dim,
        action_dim,
    )

    x = torch.randn(8, state_dim)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    print("Greedy action:", model.act(x[0]))