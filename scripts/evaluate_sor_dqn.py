"""
Evaluate a trained SOR-DQN agent.
"""

from __future__ import annotations

import numpy as np
import torch

from lane_change_rl.env.lane_change_env import LaneChangeEnv

from sor_dqn.agent import DQNAgent
from sor_dqn.config import Config

import json


def evaluate(
    checkpoint: str,
    episodes: int = 20,
):

    env = LaneChangeEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=Config,
    )

    agent.load(checkpoint)

    agent.eval()

    rewards = []

    successes = 0

    collisions = 0

    timeouts = 0

    episode_lengths = []

    episode_data = []
    action_counts = np.zeros(action_dim, dtype=int)
    episode_rewards = []
    lane_change_attempts = []
    left_episodes = 0
    right_episodes = 0
    left_success = 0
    right_success = 0

    for episode in range(episodes):

        state, info = env.reset()
        target = env.sim.target_direction
        episode_attempts = 0

        if target == -1:
            left_episodes += 1
        else:
            right_episodes += 1

        terminated = False
        truncated = False

        total_reward = 0.0

        steps = 0

        while not (terminated or truncated):

            action = agent.select_action(
                state,
                evaluate=True,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            state = next_state

            action_counts[action] += 1

            if action in [1, 2]:
                episode_attempts += 1

            steps += 1

        rewards.append(total_reward)
        episode_rewards.append(total_reward)
        lane_change_attempts.append(episode_attempts)

        episode_lengths.append(steps)

        episode_data.append(
            {
                "episode": episode + 1,
                "reward": total_reward,
                "steps": steps,
                "success": int(info.get("success", False)),
                "collision": int(info.get("collision", False)),
                "timeout": int(info.get("timeout", False)),
                "lane_changed": int(info.get("lane_changed", False)),
                "lane_invasion": int(info.get("lane_invasion", False)),
                "target": "LEFT" if target == -1 else "RIGHT",
                "attempts": episode_attempts,
            }
        )

        if info.get("success", False):
            successes += 1

            if target == -1:
                left_success += 1
            else:
                right_success += 1

        if info.get("collision", False):
            collisions += 1

        if truncated:
            timeouts += 1

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward {total_reward:8.2f} | "
            f"Steps {steps:4d}"
        )

    summary = {
        "episodes": episode_data,
        "action_counts": action_counts.tolist(),
        "metrics": {
            "episodes": episodes,
            "average_reward": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "average_episode_length": float(np.mean(episode_lengths)),
            "success_rate": float(successes / episodes),
            "collision_rate": float(collisions / episodes),
            "timeout_rate": float(timeouts / episodes),
            "average_attempts": float(np.mean(lane_change_attempts)),
            "left_success_rate":
                float(left_success / max(left_episodes, 1)),
            "right_success_rate":
                float(right_success / max(right_episodes, 1)),
        }
    }

    with open("evaluation.json", "w") as f:
        json.dump(summary, f, indent=4)

    env.close()

    print("\n================ Evaluation ================\n")

    print(f"Episodes          : {episodes}")

    print(f"Average Reward    : {np.mean(rewards):.2f}")

    print(f"Reward Std        : {np.std(rewards):.2f}")

    print(f"Average Length    : {np.mean(episode_lengths):.1f}")

    print(f"Success Rate      : {100*successes/episodes:.2f}%")

    print(f"Collision Rate    : {100*collisions/episodes:.2f}%")

    print(f"Timeout Rate      : {100*timeouts/episodes:.2f}%")

    print("\n============================================\n")


if __name__ == "__main__":

    evaluate(
        checkpoint="checkpoints/checkpoint_1000000.pth",
        episodes=500,
    )