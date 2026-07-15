"""
Reward function for lane-change RL.
"""

from __future__ import annotations

import carla
import random


class RewardFunction:
    """
    Computes reward and episode termination.
    """

    def __init__(
        self,
        world,
        max_steps: int = 1000,
    ):

        self.world = world

        self.max_steps = max_steps

        self.reset()

    # ---------------------------------------------------------
    # Episode
    # ---------------------------------------------------------

    def reset(self):

        self._previous_location = None

        self._previous_lane = None

        self._steps = 0

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------

    def compute(self):

        ego = self.world.actors.ego.vehicle

        if ego is None:
            raise RuntimeError("Ego vehicle not registered")

        # step penalty
        reward = -0.1

        terminated = False

        truncated = False

        info = {}

        # -------------------------------------------------
        # Progress reward
        # -------------------------------------------------

        location = ego.get_location()

        if self._previous_location is not None:

            delta = location.distance(
                self._previous_location
            )

            # reward += 0.1 * delta

        self._previous_location = location

        # -------------------------------------------------
        # Speed reward
        # -------------------------------------------------

        velocity = ego.get_velocity()

        speed = (
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        ) ** 0.5

        speed_limit = ego.get_speed_limit()

        # if speed_limit > 1e-3:

        #     reward += 0.05 * min(
        #         speed / speed_limit,
        #         1.0,
        #     )

        # -------------------------------------------------
        # Lane change reward
        # -------------------------------------------------

        waypoint = self.world.map.get_waypoint(
            location
        )

        lane = waypoint.lane_id

        if self._previous_lane is None:

            self._previous_lane = lane

        elif lane != self._previous_lane:

            info["lane_changed"] = True

            if lane == self.world.target_lane_id:

                reward += 200.0

                terminated = True

                info["success"] = True

            else:

                reward -= 20.0

            self._previous_lane = lane

        # -------------------------------------------------
        # Collision
        # -------------------------------------------------

        collision = (
            self.world.actors.ego.collision_sensor
        )

        if (
            collision is not None
            and collision.has_collision
        ):

            reward -= 200.0

            terminated = True

            info["collision"] = True

        # -------------------------------------------------
        # Lane invasion
        # -------------------------------------------------

        lane_sensor = (
            self.world.actors.ego.lane_invasion_sensor
        )

        if (
            lane_sensor is not None
            and lane_sensor.has_invasion
        ):

            reward -= 5.0

            info["lane_invasion"] = True

        # -------------------------------------------------
        # Timeout
        # -------------------------------------------------

        self._steps += 1

        info["lane_change_attempts"] = (
            self.world.lane_change_attempts
        )

        if self._steps >= self.max_steps:

            truncated = True

            reward -= 100.0

            info["timeout"] = True
            info["target_lane"] = self.world.target_lane_id
            info["final_lane"] = lane

        return (
            reward,
            terminated,
            truncated,
            info,
        )