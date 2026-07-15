"""
Action handler for the lane-change environment.
"""

from __future__ import annotations

from enum import IntEnum

import carla


class Action(IntEnum):

    KEEP_LANE = 0

    CHANGE_LEFT = 1

    CHANGE_RIGHT = 2


class ActionHandler:

    def __init__(
        self,
        world,
    ):

        self.world = world

        self.target_speed = 35.0  # km/h

        self.cooldown = 0

    # ---------------------------------------------------------
    # Apply Action
    # ---------------------------------------------------------

    def apply(self, action):

        ego = self.world.actors.ego.vehicle
        tm = self.world.traffic

        command_sent = False

        if self.cooldown > 0:
            self.cooldown -= 1

        else:

            if action == Action.CHANGE_LEFT:

                tm.force_lane_change(
                    ego,
                    to_right=False,
                )

                self.cooldown = 20
                self.world.lane_change_attempts += 1
                command_sent = True

            elif action == Action.CHANGE_RIGHT:

                tm.force_lane_change(
                    ego,
                    to_right=True,
                )

                self.cooldown = 20
                self.world.lane_change_attempts += 1
                command_sent = True

        speed_limit = ego.get_speed_limit()

        difference = (
            100.0 * (speed_limit - self.target_speed) / speed_limit
        )

        tm.set_speed_difference(
            ego,
            difference,
        )

        return command_sent