from lane_change_rl.env.lane_change_env import LaneChangeEnv

env = LaneChangeEnv()

obs, info = env.reset()

episode_reward = 0

while True:

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    if terminated or truncated:

        print("Episode reward:", episode_reward)

        episode_reward = 0

        obs, info = env.reset()