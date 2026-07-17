"""
agent.py

DQN / SOR-DQN Agent.
"""

import random

import numpy as np
import torch
import torch.optim as optim

from .network import QNetwork


class DQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        config,
    ):

        self.device = config.DEVICE

        self.action_dim = action_dim

        ###############################################
        # Networks
        ###############################################

        self.q_network = QNetwork(
            state_dim,
            action_dim,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim,
            action_dim,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        ###############################################
        # Optimizer
        ###############################################

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.LR,
        )

        ###############################################
        # Hyperparameters
        ###############################################

        self.gamma = config.GAMMA

        self.omega = config.OMEGA

        ###############################################
        # Epsilon-greedy
        ###############################################

        self.epsilon = config.EPS_START

        self.epsilon_start = config.EPS_START

        self.epsilon_end = config.EPS_END

        self.epsilon_decay_steps = config.EPS_DECAY_STEPS

    #########################################################
    # Action Selection
    #########################################################

    def select_action(
        self,
        state,
        evaluate=False,
    ):

        if (not evaluate) and random.random() < self.epsilon:

            return random.randrange(self.action_dim)

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q = self.q_network(state)

            action = q.argmax(dim=1).item()

        return action

    #########################################################
    # Epsilon Decay
    #########################################################

    def update_epsilon(
        self,
        current_step,
    ):

        fraction = min(
            current_step / self.epsilon_decay_steps,
            1.0,
        )

        self.epsilon = (
            self.epsilon_start
            + fraction
            * (self.epsilon_end - self.epsilon_start)
        )

    #########################################################
    # Target Update
    #########################################################

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    #########################################################
    # Save
    #########################################################

    def save(
        self,
        filename,
    ):

        checkpoint = {

            "model": self.q_network.state_dict(),

            "target": self.target_network.state_dict(),

            "optimizer": self.optimizer.state_dict(),

            "epsilon": self.epsilon,

        }

        torch.save(
            checkpoint,
            filename,
        )

    #########################################################
    # Load
    #########################################################

    def load(
        self,
        filename,
    ):

        checkpoint = torch.load(
            filename,
            map_location=self.device,
        )

        self.q_network.load_state_dict(
            checkpoint["model"]
        )

        self.target_network.load_state_dict(
            checkpoint["target"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

    #########################################################
    # Train / Eval
    #########################################################

    def train(self):

        self.q_network.train()

    def eval(self):

        self.q_network.eval()