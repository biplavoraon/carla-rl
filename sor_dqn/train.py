"""
train.py

Main training script for

- DQN
- Double DQN
- SOR-DQN
- SOR-Double DQN
"""

import os

import gym
import numpy as np
import torch

from config import Config
from replay_buffer import ReplayBuffer
from agent import DQNAgent
from learner import SORDQNTrainer


def train():

    ####################################################
    # Environment
    ####################################################

    env = gym.make(Config.ENV_NAME)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    ####################################################
    # Agent
    ####################################################

    agent = DQNAgent(
        state_dim,
        action_dim,
        Config,
    )

    ####################################################
    # Trainer
    ####################################################

    trainer = SORDQNTrainer(

        agent,

        algorithm="sor_dqn",       # change here

    )

    ####################################################
    # Replay Buffer
    ####################################################

    replay_buffer = ReplayBuffer(
        Config.BUFFER_SIZE,
        Config.DEVICE,
    )

    ####################################################
    # Logging
    ####################################################

    episode_reward = 0

    episode = 0

    state = env.reset()

    ####################################################
    # Main Loop
    ####################################################

    for step in range(Config.TOTAL_STEPS):

        ###############################################
        # Update epsilon
        ###############################################

        agent.update_epsilon(step)

        ###############################################
        # Action
        ###############################################

        action = agent.select_action(state)

        ###############################################
        # Environment step
        ###############################################

        next_state, reward, done, info = env.step(action)

        ###############################################
        # Store transition
        ###############################################

        replay_buffer.push(

            state,

            action,

            reward,

            next_state,

            done,

        )

        state = next_state

        episode_reward += reward

        ###############################################
        # Learning
        ###############################################

        if len(replay_buffer) >= Config.MIN_REPLAY_SIZE:

            loss = trainer.train_step(

                replay_buffer,

                Config.BATCH_SIZE,

            )

        ###############################################
        # Target Update
        ###############################################

        if step % Config.TARGET_UPDATE == 0:

            agent.update_target_network()

        ###############################################
        # Logging
        ###############################################

        if step % Config.LOG_INTERVAL == 0:

            print(
                f"Step {step:8d}"
                f" | Buffer {len(replay_buffer):6d}"
                f" | Epsilon {agent.epsilon:.3f}"
            )

        ###############################################
        # Save
        ###############################################

        if step % Config.SAVE_INTERVAL == 0:

            os.makedirs(
                Config.CHECKPOINT_DIR,
                exist_ok=True,
            )

            agent.save(
                os.path.join(
                    Config.CHECKPOINT_DIR,
                    f"checkpoint_{step}.pth",
                )
            )

        ###############################################
        # Episode End
        ###############################################

        if done:

            print(
                f"Episode {episode:5d}"
                f" Reward {episode_reward:8.2f}"
            )

            episode += 1

            episode_reward = 0

            state = env.reset()

    env.close()


if __name__ == "__main__":

    train()