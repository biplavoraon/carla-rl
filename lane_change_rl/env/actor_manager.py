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
import time


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

        self._valid_spawn_points = []

        for spawn in self._spawn_points:

            wp = self._world.get_map().get_waypoint(spawn.location)

            left = wp.get_left_lane()
            right = wp.get_right_lane()

            if (
                left is not None and left.lane_type == carla.LaneType.Driving
            ) or (
                right is not None and right.lane_type == carla.LaneType.Driving
            ):
                self._valid_spawn_points.append(spawn)

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


    def reset_ego(self):

        spawn = random.choice(self._spawn_points)

        ego = self._ego.vehicle

        # print(
        #     "alive:",
        #     ego.is_alive,
        #     "id:",
        #     ego.id,
        # )

        ego.set_transform(spawn)

        ego.set_target_velocity(
            carla.Vector3D()
        )

        ego.set_target_angular_velocity(
            carla.Vector3D()
        )

        ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
            )
        )

    # ---------------------------------------------------------
    # Ego Vehicle
    # ---------------------------------------------------------

    def spawn_ego(self) -> EgoVehicle:
        """
        Spawn the ego vehicle on a lane that has at least one
        adjacent driving lane.
        """

        if self._ego is not None:
            raise SpawnError("Ego vehicle already exists.")

        blueprint = self._blueprints.find(
            self._cfg.ego.blueprint
        )

        spawn_points = list(self._valid_spawn_points)
        random.shuffle(spawn_points)

        vehicle = None

        for spawn_point in spawn_points:

            actor = self._world.try_spawn_actor(
                blueprint,
                spawn_point,
            )

            if actor is None:
                continue

            waypoint = self._world.get_map().get_waypoint(
                actor.get_location()
            )

            if waypoint.lane_type != carla.LaneType.Driving:
                actor.destroy()
                continue

            left = waypoint.get_left_lane()
            right = waypoint.get_right_lane()

            left_valid = (
                left is not None
                and left.lane_type == carla.LaneType.Driving
            )

            right_valid = (
                right is not None
                and right.lane_type == carla.LaneType.Driving
            )

            # Accept only spawn points from which a lane change
            # is actually possible.
            if left_valid or right_valid:

                vehicle = actor
                break

            # Reject this spawn point
            actor.destroy()

        if vehicle is None:
            raise SpawnError(
                "Unable to spawn ego vehicle on a valid lane."
            )

        self._traffic_manager.register_ego_vehicle(
            vehicle
        )

        self._ego = EgoVehicle(
            vehicle=vehicle,
        )

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

    # ---------------------------------------------------------
    # Ego
    # ---------------------------------------------------------

    def destroy_ego(self):

        if self._ego is None:
            return

        try:
            self._ego.destroy()
        except RuntimeError:
            pass

        self._ego = None

        self._logger.info(
            "Destroyed ego vehicle."
        )

    # ---------------------------------------------------------
    # Background Traffic
    # ---------------------------------------------------------

    def destroy_background_traffic(self) -> None:
        """
        Destroy all background traffic.
        """

        if not self._traffic:
            return

        batch = [
            command.DestroyActor(vehicle)
            for vehicle in self._traffic
            if vehicle.is_alive
        ]

        if batch:
            self._client.apply_batch_sync(
                batch,
                True,
            )

        destroyed = len(self._traffic)

        self._traffic.clear()

        self._logger.info(
            "Destroyed %d traffic vehicles.",
            destroyed,
        )

    # ---------------------------------------------------------
    # Sensors
    # ---------------------------------------------------------

    def destroy_sensors(self) -> None:
        """
        Destroy standalone sensors.
        """

        batch = [
            command.DestroyActor(sensor)
            for sensor in self._sensors
            if sensor.is_alive
        ]

        if batch:
            self._client.apply_batch_sync(
                batch,
                True,
            )

        self._sensors.clear()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def destroy_all(self):

        self.destroy_sensors()

        self.destroy_background_traffic()

        self.destroy_ego()

        self._traffic.clear()
        self._walkers.clear()
        self._sensors.clear()
        self._ego = None

        self._logger.info(
            "Actor cleanup complete."
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def random_spawn_point(self) -> carla.Transform:
        """
        Return a random spawn point.
        """

        return random.choice(
            self._spawn_points
        )


    def available_spawn_points(self):
        """
        Return all spawn points.
        """

        return tuple(self._spawn_points)


    def number_of_traffic(self) -> int:

        return len(self._traffic)


    def has_ego(self) -> bool:

        return self._ego is not None

    
