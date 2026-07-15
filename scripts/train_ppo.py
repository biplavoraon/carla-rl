from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import lane_change_rl.register
from lane_change_rl.env.lane_change_env import LaneChangeEnv

Path("models/checkpoints").mkdir(
    parents=True,
    exist_ok=True,
)

Path("logs").mkdir(
    exist_ok=True,
)

train_env = Monitor(
    LaneChangeEnv()
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="models/checkpoints",
    name_prefix="ppo_lane_change",
)

# -----------------------------------------
# Resume training
# -----------------------------------------

model = PPO.load(
    "models/final_model",
    env=train_env,
)

try:

    model.learn(
        total_timesteps=2_500_000,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

finally:

    model.save(
        "models/final_model_v2",
    )

    train_env.close()