from gymnasium.envs.registration import register

register(
    id="LaneChange-v0",
    entry_point="lane_change_rl.env.lane_change_env:LaneChangeEnv",
)