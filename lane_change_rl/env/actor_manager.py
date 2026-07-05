"""
Actor manager.

Responsible for:
    - Spawning the ego vehicle
    - Spawning background traffic
    - Destroying actors
    - Keeping references to all actors
"""

from __future__ import annotations

import random

import carla
from carla import command

from lane_change_rl.models import EgoVehicle
from lane_change_rl.env.exceptions import SpawnError
from lane_change_rl.env.traffic_manager import TrafficManager
from lane_change_rl.config.loader import Config


class ActorManager:

    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        traffic_manager: TrafficManager,
        cfg: Config,
        logger,
    ):

        self._client = client
        self._world = world
        self._blueprints = blueprint_library

        self._traffic_manager = traffic_manager

        self._cfg = cfg

        self._logger = logger

        self._ego: EgoVehicle | None = None

        self._traffic: list[carla.Vehicle] = []

        self._walkers = []

        self._sensors = []

        self._spawn_points = (
            self._world.get_map().get_spawn_points()
        )

        random.seed(cfg.traffic.seed)

        @property
    def ego(self) -> EgoVehicle | None:
        return self._ego

    @property
    def ego_vehicle(self) -> carla.Vehicle | None:
        if self._ego is None:
            return None

        return self._ego.vehicle

    @property
    def traffic(self):
        return tuple(self._traffic)

    @property
    def actors(self):
        """
        Return every actor managed by this class.
        """

        actors = []

        if self._ego is not None:
            actors.append(self._ego.vehicle)

        actors.extend(self._traffic)

        actors.extend(self._sensors)

        actors.extend(self._walkers)

        return actors

        def reset(self):

        self.destroy_all()

        self._ego = None

        self._traffic.clear()

        self._walkers.clear()

        self._sensors.clear()

    # ---------------------------------------------------------
# Ego Vehicle
# ---------------------------------------------------------

def spawn_ego(self) -> EgoVehicle:
    """
    Spawn the ego vehicle.
    """

    if self._ego is not None:
        raise SpawnError("Ego vehicle already exists.")

    blueprint = self._blueprints.find(
        self._cfg.ego.blueprint
    )

    if self._cfg.ego.spawn.random:

        spawn_point = random.choice(
            self._spawn_points
        )

    else:

        idx = self._cfg.ego.spawn.spawn_index

        spawn_point = self._spawn_points[idx]

    vehicle = self._world.try_spawn_actor(
        blueprint,
        spawn_point,
    )

    if vehicle is None:
        raise SpawnError(
            "Unable to spawn ego vehicle."
        )

    self._ego = EgoVehicle(vehicle=vehicle)

    self._logger.info(
        "Spawned ego vehicle."
    )

    return self._ego

    # ---------------------------------------------------------
# Background Traffic
# ---------------------------------------------------------

def spawn_background_traffic(self) -> int:
    """
    Spawn background traffic using CARLA's batch API.

    Returns
    -------
    int
        Number of successfully spawned vehicles.
    """

    spawn_points = list(self._spawn_points)

    random.shuffle(spawn_points)

    blueprints = self._blueprints.filter(
        "vehicle.*"
    )

    batch = []

    count = min(
        self._cfg.traffic.num_vehicles,
        len(spawn_points),
    )

    for spawn_point in spawn_points[:count]:

        blueprint = random.choice(
            blueprints
        )

        if blueprint.has_attribute("color"):

            color = random.choice(
                blueprint.get_attribute("color").recommended_values
            )

            blueprint.set_attribute(
                "color",
                color,
            )

        if blueprint.has_attribute("driver_id"):

            driver = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )

            blueprint.set_attribute(
                "driver_id",
                driver,
            )

        blueprint.set_attribute(
            "role_name",
            "autopilot",
        )

        batch.append(

            command.SpawnActor(
                blueprint,
                spawn_point,
            ).then(

                command.SetAutopilot(
                    command.FutureActor,
                    True,
                    self._traffic_manager.port,
                )

            )

        )

    responses = self._client.apply_batch_sync(
        batch,
        True,
    )

    spawned = 0

    for response in responses:

        if response.error:
            continue

        actor = self._world.get_actor(
            response.actor_id
        )

        if actor is None:
            continue

        self._traffic.append(actor)

        self._traffic_manager.register_background_vehicle(
            actor
        )

        spawned += 1

    self._logger.info(
        "Spawned %d traffic vehicles.",
        spawned,
    )

    return spawned