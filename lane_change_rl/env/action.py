"""
Discrete action space for lane change RL.
"""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """
    High-level actions available to the RL agent.
    """

    KEEP_LANE = 0

    CHANGE_LEFT = 1

    CHANGE_RIGHT = 2

    ACCELERATE = 3

    DECELERATE = 4

    BRAKE = 5