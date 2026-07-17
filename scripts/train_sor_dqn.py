"""
CARLA SOR-DQN Training Script

Compatible with:
    - DQN
    - Double DQN
    - SOR-DQN
    - SOR-Double DQN
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lane_change_rl.env.lane_change_env import LaneChangeEnv

from sor_dqn.agent import DQNAgent
from sor_dqn.replay_buffer import ReplayBuffer
from sor_dqn.learner import SORDQNTrainer
from sor_dqn.config import Config


################################################################################
# Utilities
################################################################################


def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


################################################################################
# Checkpoint
################################################################################


def save_checkpoint(agent, step):

    os.makedirs(
        Config.CHECKPOINT_DIR,
        exist_ok=True,
    )

    filename = os.path.join(
        Config.CHECKPOINT_DIR,
        f"checkpoint_{step}.pth",
    )

    agent.save(filename)

    print(f"\nCheckpoint saved -> {filename}")



################################################################################
# TensorBoard
################################################################################


writer = SummaryWriter(
    log_dir=Config.LOG_DIR,
)


################################################################################
# Training
################################################################################


def train():

    set_seed(Config.SEED)

    env = LaneChangeEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 70)
    print("State dimension :", state_dim)
    print("Action dimension:", action_dim)
    print("Device          :", Config.DEVICE)
    print("=" * 70)

    ############################################################
    # Agent
    ############################################################

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=Config,
    )

    ############################################################
    # Trainer
    ############################################################

    trainer = SORDQNTrainer(
        agent,
        algorithm="sor_dqn",      # dqn / double_dqn / sor_dqn / sor_double_dqn
    )

    ############################################################
    # Replay Buffer
    ############################################################

    replay_buffer = ReplayBuffer(
        capacity=Config.BUFFER_SIZE,
        device=Config.DEVICE,
    )

    ############################################################
    # Statistics
    ############################################################

    global_step = 0

    episode = 0

    best_reward = -1e9

    progress = tqdm(total=Config.TOTAL_STEPS)

    ############################################################
    # Training Loop
    ############################################################

    while global_step < Config.TOTAL_STEPS:

        state, _ = env.reset()

        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0

        while not (terminated or truncated):

            ####################################################
            # epsilon schedule
            ####################################################

            agent.update_epsilon(global_step)

            ####################################################
            # action
            ####################################################

            action = agent.select_action(state)

            ####################################################
            # environment
            ####################################################

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            done = terminated or truncated

            ####################################################
            # replay buffer
            ####################################################

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            episode_reward += reward

            episode_steps += 1

            global_step += 1

            progress.update(1)

            ####################################################
            # Learn
            ####################################################

            if len(replay_buffer) >= Config.MIN_REPLAY_SIZE:

                loss = trainer.train_step(
                    replay_buffer,
                    Config.BATCH_SIZE,
                )

                writer.add_scalar(
                    "train/loss",
                    loss,
                    global_step,
                )

            ####################################################
            # Target Network
            ####################################################

            if global_step % Config.TARGET_UPDATE == 0:

                agent.update_target_network()

            
            ####################################################
            # Periodic checkpoint
            ####################################################

            if global_step % Config.SAVE_INTERVAL == 0:

                save_checkpoint(
                    agent,
                    global_step,
                )

            ####################################################
            # Stop training
            ####################################################

            if global_step >= Config.TOTAL_STEPS:
                break

        ########################################################
        # Episode finished
        ########################################################

        writer.add_scalar(
            "train/episode_reward",
            episode_reward,
            episode,
        )

        writer.add_scalar(
            "train/episode_length",
            episode_steps,
            episode,
        )

        writer.add_scalar(
            "train/epsilon",
            agent.epsilon,
            episode,
        )

        print(
            f"Episode {episode:5d} | "
            f"Reward {episode_reward:8.2f} | "
            f"Length {episode_steps:4d} | "
            f"Epsilon {agent.epsilon:.3f}"
        )

        episode += 1

    ############################################################

    progress.close()

    writer.close()

    env.close()

    print("\nTraining completed.")


################################################################################
# Main
################################################################################


def main():

    print("=" * 70)
    print("SOR-DQN Trainer")
    print("=" * 70)

    Path(Config.LOG_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(Config.CHECKPOINT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    train()


if __name__ == "__main__":

    main()