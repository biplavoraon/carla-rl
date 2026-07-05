"""
Configuration loader.

Loads the YAML configuration file and converts it into strongly typed
dataclasses so the rest of the codebase never deals with nested dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CarlaConfig:
    host: str
    port: int
    timeout: float


@dataclass(frozen=True)
class SimulationConfig:
    synchronous_mode: bool
    fixed_delta_seconds: float
    fps: int


@dataclass(frozen=True)
class MapConfig:
    town: str


@dataclass(frozen=True)
class EgoSpawnConfig:
    random: bool
    spawn_index: int


@dataclass(frozen=True)
class EgoConfig:
    blueprint: str
    spawn: EgoSpawnConfig


@dataclass(frozen=True)
class TrafficConfig:
    tm_port: int
    seed: int
    num_vehicles: int
    hybrid_physics: bool
    global_speed_difference: float
    auto_lane_change: bool


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fov: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CollisionSensorConfig:
    enabled: bool


@dataclass(frozen=True)
class LaneInvasionSensorConfig:
    enabled: bool


@dataclass(frozen=True)
class EnvironmentConfig:
    max_episode_steps: int


@dataclass(frozen=True)
class SeedConfig:
    python: int
    numpy: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    save_directory: str


@dataclass(frozen=True)
class DebugConfig:
    spectator_follow: bool
    draw_waypoints: bool
    draw_spawn_points: bool
    verbose: bool


@dataclass(frozen=True)
class Config:
    carla: CarlaConfig
    simulation: SimulationConfig
    map: MapConfig
    ego: EgoConfig
    traffic: TrafficConfig
    camera: CameraConfig
    collision_sensor: CollisionSensorConfig
    lane_invasion_sensor: LaneInvasionSensorConfig
    environment: EnvironmentConfig
    seed: SeedConfig
    logging: LoggingConfig
    debug: DebugConfig


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """
    Load YAML configuration.

    Parameters
    ----------
    path : str | Path
        Path to config.yaml.

    Returns
    -------
    Config
        Parsed configuration object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    return Config(
        carla=CarlaConfig(**data["carla"]),
        simulation=SimulationConfig(**data["simulation"]),
        map=MapConfig(**data["map"]),
        ego=EgoConfig(
            blueprint=data["ego"]["blueprint"],
            spawn=EgoSpawnConfig(**data["ego"]["spawn"]),
        ),
        traffic=TrafficConfig(**data["traffic"]),
        camera=CameraConfig(**data["camera"]),
        collision_sensor=CollisionSensorConfig(
            **data["collision_sensor"]
        ),
        lane_invasion_sensor=LaneInvasionSensorConfig(
            **data["lane_invasion_sensor"]
        ),
        environment=EnvironmentConfig(**data["environment"]),
        seed=SeedConfig(**data["seed"]),
        logging=LoggingConfig(**data["logging"]),
        debug=DebugConfig(**data["debug"]),
    )