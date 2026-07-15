from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lane_change_rl.env.action import ActionHandler
from lane_change_rl.env.observation import ObservationExtractor
from lane_change_rl.env.reward import RewardFunction
from lane_change_rl.env.world import CarlaWorld


class LaneChangeEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20,
    }

    def __init__(self):

        super().__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        self.sim = CarlaWorld()

        self.sim.initialize()

        self.observer = ObservationExtractor(self.sim)

        self.reward_fn = RewardFunction(self.sim)

        self.action_handler = ActionHandler(self.sim)

    # -----------------------------------------------------

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.sim.reset_episode()

        self.reward_fn.reset()

        observation = self.observer.observe()

        return observation, {}

    # -----------------------------------------------------

    def step(self, action):

        command_sent = self.action_handler.apply(action)

        self.sim.tick()

        observation = self.observer.observe()

        reward, terminated, truncated, info = (
            self.reward_fn.compute()
        )

        info["lane_change_command"] = command_sent

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------

    def render(self):

        pass

    # -----------------------------------------------------

    def close(self):

        if self.sim is not None:
            self.sim.close()
            self.sim = None