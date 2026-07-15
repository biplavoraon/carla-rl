"""
CARLA World.

Responsible for:

    - Connecting to CARLA
    - Loading maps
    - Simulation settings
    - Creating managers

This class DOES NOT spawn vehicles.
"""

from __future__ import annotations

import carla

from lane_change_rl.config import load_config
from lane_change_rl.env.exceptions import CarlaConnectionError
from lane_change_rl.env.traffic_manager import TrafficManager
from lane_change_rl.env.utils import setup_logger, set_random_seed
from lane_change_rl.env.actor_manager import ActorManager

import random


class CarlaWorld:
    """
    Main interface to the CARLA simulator.
    """

    def __init__(self):

        self.cfg = load_config()

        self.logger = setup_logger(
            self.__class__.__name__,
            self.cfg.logging.level,
        )

        self.target_direction = 0

        self.target_lane_id = None

        self.lane_change_attempts = 0

        # ---------------------------------------------------------
        # Random seed
        # ---------------------------------------------------------

        set_random_seed(
            self.cfg.seed.python
        )

        # ---------------------------------------------------------
        # Connect to CARLA
        # ---------------------------------------------------------

        try:

            self.client = carla.Client(
                self.cfg.carla.host,
                self.cfg.carla.port,
            )

            self.client.set_timeout(
                self.cfg.carla.timeout
            )

        except Exception as e:

            raise CarlaConnectionError(
                "Unable to connect to CARLA."
            ) from e

        # ---------------------------------------------------------
        # Load world
        # ---------------------------------------------------------

        current_world = self.client.get_world()

        current_map = current_world.get_map().name.split("/")[-1]

        if current_map != self.cfg.map.town:

            # print(f"Loading map {self.cfg.map.town}")

            self.client.load_world(
                self.cfg.map.town,
                reset_settings=False,
            )

            self.world = self.client.get_world()

        else:

            # print("Reusing existing world")

            self.world = current_world

        self.map = self.world.get_map()

        self.blueprints = (
            self.world.get_blueprint_library()
        )

        # ---------------------------------------------------------
        # Simulation settings
        # ---------------------------------------------------------

        self.original_settings = (
            self.world.get_settings()
        )

        self._apply_settings()


        # ---------------------------------------------------------
        # Managers
        # ---------------------------------------------------------

        self.traffic = TrafficManager(
            self.client,
            self.cfg.traffic,
        )

        # These will be initialized later

        self.actors = ActorManager(
            client=self.client,
            world=self.world,
            blueprint_library=self.blueprints,
            traffic_manager=self.traffic,
            cfg=self.cfg,
            logger=self.logger,
        )

        self.maps = None

        self.debug = None

        self.logger.info(
            "Loaded map: %s",
            self.map.name,
        )

    # ---------------------------------------------------------
    # Settings
    # ---------------------------------------------------------

    def _apply_settings(self):

        settings = self.world.get_settings()

        settings.synchronous_mode = (
            self.cfg.simulation.synchronous_mode
        )

        settings.fixed_delta_seconds = (
            self.cfg.simulation.fixed_delta_seconds
        )

        self.world.apply_settings(
            settings
        )

    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------


    def initialize(self):

        self.actors.spawn_ego()

        self.tick()

        self.actors.spawn_background_traffic()

        self.tick()


    def tick(self):

        return self.world.tick()



    def reset_episode(self):

        self.actors.reset()

        ego = self.actors.spawn_ego()

        self.tick()

        self.follow_ego()

        self.actors.spawn_background_traffic()

        self.tick()

        self.lane_change_attempts = 0

        waypoint = self.map.get_waypoint(
            ego.vehicle.get_location()
        )

        # print("Current lane:", waypoint.lane_id)

        targets = []

        left = waypoint.get_left_lane()

        if (left is not None and left.lane_type == carla.LaneType.Driving):
            targets.append(left)

        right = waypoint.get_right_lane()

        if (right is not None and right.lane_type == carla.LaneType.Driving):
            targets.append(right)

        # print("Left: ", left)
        # print("Right:", right)

        # if left is not None:
        #     print("Left lane id:", left.lane_type)

        # if right is not None:
        #     print("Right lane id:", right.lane_type)

        if not targets:
            raise RuntimeError(
                "No adjacent lane available."
            )

        target_wp = random.choice(targets)

        self.target_lane_id = target_wp.lane_id

        if (
            left is not None
            and target_wp.lane_id == left.lane_id
        ):
            self.target_direction = -1
        else:
            self.target_direction = 1


    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    # def close(self):

    #     if self.actors is not None:
    #         self.actors.destroy_all()

    #     self.world.apply_settings(
    #         self.original_settings
    #     )

    #     self.logger.info(
    #         "World closed."
    #     )


    def follow_ego(
        self,
        distance: float = 6.0,
        height: float = 2.5,
        pitch: float = -10.0,
    ) -> None:
        """
        Position the spectator behind the ego vehicle.
        """

        ego = self.actors.ego_vehicle

        if ego is None or not ego.is_alive:
            return

        transform = ego.get_transform()

        spectator = self.world.get_spectator()

        spectator.set_transform(
            carla.Transform(
                transform.transform(
                    carla.Location(
                        x=-distance,
                        z=height,
                    )
                ),
                carla.Rotation(
                    pitch=pitch,
                    yaw=transform.rotation.yaw,
                    roll=0.0,
                ),
            )
        )

    def close(self):

        if self.actors is not None:
            self.actors.destroy_all()

        # Give CARLA one tick to process destruction
        self.world.tick()

        self.world.apply_settings(
            self.original_settings
        )

        self.logger.info("World closed.")

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        self.close()