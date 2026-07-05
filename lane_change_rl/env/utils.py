"""
Common utility functions used throughout the project.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import carla
import numpy as np


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def setup_logger(
    name: str,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create and return a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger


# ---------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """
    Set Python and NumPy random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------


def vehicle_speed(vehicle: carla.Vehicle) -> float:
    """
    Returns vehicle speed in km/h.
    """

    vel = vehicle.get_velocity()

    speed = math.sqrt(
        vel.x**2 +
        vel.y**2 +
        vel.z**2
    )

    return 3.6 * speed


def vehicle_speed_mps(vehicle: carla.Vehicle) -> float:
    """
    Returns vehicle speed in m/s.
    """

    vel = vehicle.get_velocity()

    return math.sqrt(
        vel.x**2 +
        vel.y**2 +
        vel.z**2
    )


# ---------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------


def distance_between(
    actor1: carla.Actor,
    actor2: carla.Actor,
) -> float:
    """
    Euclidean distance between two actors.
    """

    return actor1.get_location().distance(
        actor2.get_location()
    )


def distance_locations(
    loc1: carla.Location,
    loc2: carla.Location,
) -> float:
    """
    Euclidean distance between two locations.
    """

    return loc1.distance(loc2)


# ---------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-180, 180].
    """

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------


def transform_to_numpy(
    transform: carla.Transform,
) -> np.ndarray:
    """
    Transform -> numpy array.
    """

    loc = transform.location

    rot = transform.rotation

    return np.array(
        [
            loc.x,
            loc.y,
            loc.z,
            rot.roll,
            rot.pitch,
            rot.yaw,
        ],
        dtype=np.float32,
    )


def location_to_numpy(
    location: carla.Location,
) -> np.ndarray:
    """
    Convert location to numpy.
    """

    return np.array(
        [
            location.x,
            location.y,
            location.z,
        ],
        dtype=np.float32,
    )


def vector3d_to_numpy(
    vector: carla.Vector3D,
) -> np.ndarray:
    """
    Convert Vector3D to numpy.
    """

    return np.array(
        [
            vector.x,
            vector.y,
            vector.z,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clamp a value.
    """

    return max(
        minimum,
        min(value, maximum),
    )


def heading_error(
    vehicle: carla.Vehicle,
    waypoint: carla.Waypoint,
) -> float:
    """
    Difference between vehicle heading and lane heading.
    """

    vehicle_yaw = vehicle.get_transform().rotation.yaw

    lane_yaw = waypoint.transform.rotation.yaw

    return normalize_angle(
        vehicle_yaw - lane_yaw
    )


# ---------------------------------------------------------------------
# Spectator
# ---------------------------------------------------------------------


def follow_vehicle(
    world: carla.World,
    vehicle: carla.Vehicle,
    distance: float = 8.0,
    height: float = 3.0,
):
    """
    Third-person chase camera.
    """

    transform = vehicle.get_transform()

    forward = transform.get_forward_vector()

    location = transform.location - forward * distance

    location.z += height

    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            location,
            transform.rotation,
        )
    )


def top_down_view(
    world: carla.World,
    vehicle: carla.Vehicle,
    height: float = 35.0,
):
    """
    Bird's-eye spectator.
    """

    transform = vehicle.get_transform()

    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(z=height),
            carla.Rotation(
                pitch=-90
            ),
        )
    )


# ---------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------


def ensure_directory(path: str | Path):
    """
    Create directory if it doesn't exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )