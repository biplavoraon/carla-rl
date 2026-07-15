from pathlib import Path
import numpy as np
import json

from stable_baselines3 import PPO

import lane_change_rl.register
from lane_change_rl.env.lane_change_env import LaneChangeEnv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = "models/final_model"
# MODEL_PATH = "models/checkpoints/ppo_lane_change_300000_steps"

NUM_EPISODES = 500

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

assert Path(MODEL_PATH + ".zip").exists(), f"{MODEL_PATH}.zip not found"

model = PPO.load(MODEL_PATH)

env = LaneChangeEnv()

episode_rewards = []
episode_lengths = []

collisions = 0
timeouts = 0
lane_changes = 0
lane_invasions = 0
successes = 0
action_counts = np.zeros(3, dtype=int)
lane_change_attempts = []
left_episodes = 0
right_episodes = 0
left_success = 0
right_success = 0
episode_data = []

try:

    for episode in range(NUM_EPISODES):

        obs, info = env.reset()

        target = env.sim.target_direction

        if target == -1:
            left_episodes += 1
        else:
            right_episodes += 1

        done = False

        total_reward = 0.0
        steps = 0
        episode_attempts = 0


        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward
            steps += 1

            action_counts[action] += 1

            if action in [1, 2]:
                episode_attempts += 1

            if info.get("collision", False):
                collisions += 1

            if info.get("timeout", False):
                timeouts += 1

                print(
                    f"Timeout | "
                    f"Target={info.get('target_lane')} "
                    f"Final={info.get('final_lane')} "
                    f"Attempts={episode_attempts}"
                )

            if info.get("lane_changed", False):
                lane_changes += 1

            if info.get("lane_invasion", False):
                lane_invasions += 1

            if info.get("success", False):

                successes += 1

                if target == -1:
                    left_success += 1
                else:
                    right_success += 1

            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        lane_change_attempts.append(episode_attempts)

        episode_data.append({
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "success": int(info.get("success", False)),
            "collision": int(info.get("collision", False)),
            "timeout": int(info.get("timeout", False)),
            "lane_changed": int(info.get("lane_changed", False)),
            "lane_invasion": int(info.get("lane_invasion", False)),
            "target": "LEFT" if target == -1 else "RIGHT",
            "attempts": episode_attempts
        })

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward = {total_reward:8.2f} | "
            f"Steps = {steps:4d}"
        )

finally:

    env.close()

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

print("\n================ Evaluation Summary ================")

print(f"Episodes               : {NUM_EPISODES}")

print()

print(f"Average Reward         : {np.mean(episode_rewards):.2f}")
print(f"Reward Std             : {np.std(episode_rewards):.2f}")

print()

print(f"Average Episode Length : {np.mean(episode_lengths):.1f}")

print()

print(f"Collision Rate         : {100 * collisions / NUM_EPISODES:.2f}%")
print(f"Timeout Rate           : {100 * timeouts / NUM_EPISODES:.2f}%")

print()

print(f"Lane Changes           : {lane_changes}")
print(
    f"Average Attempts      : "
    f"{np.mean(lane_change_attempts):.2f}"
)
print(f"Lane Invasions         : {lane_invasions}")

print(
    f"Success Rate : "
    f"{100*successes/NUM_EPISODES:.2f}%"
)

total_actions = action_counts.sum()

print()

print(
    f"KEEP  : {action_counts[0]} "
    f"({100*action_counts[0]/total_actions:.1f}%)"
)

print(
    f"LEFT  : {action_counts[1]} "
    f"({100*action_counts[1]/total_actions:.1f}%)"
)

print(
    f"RIGHT : {action_counts[2]} "
    f"({100*action_counts[2]/total_actions:.1f}%)"
)

print()

print(
    "LEFT Success Rate     : "
    f"{100*left_success/max(left_episodes,1):.2f}%"
)

print(
    "RIGHT Success Rate    : "
    f"{100*right_success/max(right_episodes,1):.2f}%"
)


print("====================================================")

with open("evaluation.json", "w") as f:
    json.dump(
        {
            "episodes": episode_data,
            "action_counts": action_counts.tolist(),
        },
        f,
        indent=4,
    )