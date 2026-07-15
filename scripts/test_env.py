import gymnasium as gym

import lane_change_rl.register


env = gym.make("LaneChange-v0")

NUM_EPISODES = 5

for episode in range(NUM_EPISODES):

    obs, info = env.reset()

    print(obs)
    print(
        "Target:",
        env.unwrapped.sim.target_direction,
    )

    # Access the underlying environment because gym.make wraps it
    target = env.unwrapped.sim.target_direction

    print(f"\nEpisode {episode + 1}")

    print(
        "Target:",
        "LEFT" if target == -1 else "RIGHT",
    )

    episode_reward = 0.0

    step = 0

    while True:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

        step += 1

        if terminated or truncated:

            print(f"Steps          : {step}")
            print(f"Reward         : {episode_reward:.2f}")
            print(f"Success        : {info.get('success', False)}")
            print(f"Collision      : {info.get('collision', False)}")
            print(f"Timeout        : {info.get('timeout', False)}")
            print(f"Lane Changed   : {info.get('lane_changed', False)}")
            print(f"Lane Invasion  : {info.get('lane_invasion', False)}")

            break

env.close()

print("\nDone")