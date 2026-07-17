"""
replay_buffer.py

Experience Replay Buffer for DQN/SOR-DQN.
"""

from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    """
    Standard Experience Replay Buffer.

    Stores transitions of the form

        (state,
         action,
         reward,
         next_state,
         done)
    """

    def __init__(
        self,
        capacity: int,
        device="cpu",
    ):

        self.capacity = capacity
        self.device = device

        self.buffer = deque(maxlen=capacity)

    def __len__(self):

        return len(self.buffer)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store one transition.
        """

        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(
        self,
        batch_size,
    ):
        """
        Returns

            state
            action
            reward
            next_state
            done

        as torch tensors.
        """

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def clear(self):

        self.buffer.clear()

    def save(self, filename):
        """
        Save replay buffer to disk.
        """

        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        """
        Restore replay buffer.
        """

        import pickle

        with open(filename, "rb") as f:
            self.buffer = pickle.load(f)