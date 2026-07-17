"""
sor_dqn.py

Learning rules for

1. DQN
2. Double DQN
3. SOR-DQN
4. SOR-Double DQN

The Agent owns the networks and optimizer.
This file only computes one gradient update.
"""

import torch
import torch.nn.functional as F


class SORDQNTrainer:

    def __init__(self, agent, algorithm="sor_dqn"):

        self.agent = agent
        self.algorithm = algorithm.lower()

    ##########################################################
    # Main Optimization Step
    ##########################################################

    def train_step(
        self,
        replay_buffer,
        batch_size,
    ):

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay_buffer.sample(batch_size)

        ######################################################
        # Current Q-values
        ######################################################

        current_q = self.agent.q_network(states)

        current_q = current_q.gather(1, actions)

        ######################################################
        # Compute Target
        ######################################################

        with torch.no_grad():

            target = self.compute_target(
                states,
                rewards,
                next_states,
                dones,
            )

        ######################################################
        # Loss
        ######################################################

        loss = F.smooth_l1_loss(
            current_q,
            target,
        )

        ######################################################
        # Gradient Step
        ######################################################

        self.agent.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.agent.q_network.parameters(),
            10.0,
        )

        self.agent.optimizer.step()

        return loss.item()

    ##########################################################
    # Target Computation
    ##########################################################

    def compute_target(
        self,
        states,
        rewards,
        next_states,
        dones,
    ):

        if self.algorithm == "dqn":

            return self._dqn_target(
                rewards,
                next_states,
                dones,
            )

        elif self.algorithm == "double_dqn":

            return self._double_dqn_target(
                rewards,
                next_states,
                dones,
            )

        elif self.algorithm == "sor_dqn":

            return self._sor_target(
                states,
                rewards,
                next_states,
                dones,
            )

        elif self.algorithm == "sor_double_dqn":

            return self._sor_double_target(
                states,
                rewards,
                next_states,
                dones,
            )

        else:

            raise ValueError(
                f"Unknown algorithm {self.algorithm}"
            )

    ##########################################################
    # DQN
    ##########################################################

    def _dqn_target(
        self,
        rewards,
        next_states,
        dones,
    ):

        next_q = self.agent.target_network(next_states)

        max_next_q = next_q.max(
            dim=1,
            keepdim=True,
        )[0]

        target = (
            rewards
            + self.agent.gamma
            * (1 - dones)
            * max_next_q
        )

        return target

    ##########################################################
    # Double DQN
    ##########################################################

    def _double_dqn_target(
        self,
        rewards,
        next_states,
        dones,
    ):

        next_action = self.agent.q_network(
            next_states
        ).argmax(
            dim=1,
            keepdim=True,
        )

        max_next_q = self.agent.target_network(
            next_states
        ).gather(
            1,
            next_action,
        )

        target = (
            rewards
            + self.agent.gamma
            * (1 - dones)
            * max_next_q
        )

        return target

    ##########################################################
    # SOR-DQN
    ##########################################################

    def _sor_target(
        self,
        states,
        rewards,
        next_states,
        dones,
    ):

        next_q = self.agent.target_network(
            next_states
        )

        max_next_q = next_q.max(
            dim=1,
            keepdim=True,
        )[0]

        current_q = self.agent.target_network(
            states
        )

        max_current_q = current_q.max(
            dim=1,
            keepdim=True,
        )[0]

        bellman = (
            rewards
            + self.agent.gamma
            * (1 - dones)
            * max_next_q
        )

        target = (
            self.agent.omega * bellman
            + (1.0 - self.agent.omega)
            * max_current_q
        )

        target = torch.where(
            dones.bool(),
            rewards,
            target,
        )

        return target

    ##########################################################
    # SOR-Double DQN
    ##########################################################

    def _sor_double_target(
        self,
        states,
        rewards,
        next_states,
        dones,
    ):

        next_action = self.agent.q_network(
            next_states
        ).argmax(
            dim=1,
            keepdim=True,
        )

        max_next_q = self.agent.target_network(
            next_states
        ).gather(
            1,
            next_action,
        )

        current_q = self.agent.target_network(
            states
        )

        max_current_q = current_q.max(
            dim=1,
            keepdim=True,
        )[0]

        bellman = (
            rewards
            + self.agent.gamma
            * (1 - dones)
            * max_next_q
        )

        target = (
            self.agent.omega * bellman
            + (1.0 - self.agent.omega)
            * max_current_q
        )

        target = torch.where(
            dones.bool(),
            rewards,
            target,
        )

        return target