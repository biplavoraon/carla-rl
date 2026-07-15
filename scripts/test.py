from stable_baselines3.common.env_checker import check_env

from lane_change_rl.env.lane_change_env import LaneChangeEnv

env = LaneChangeEnv()

check_env(
    env,
    warn=True,
)

print("Environment OK")

env.close()