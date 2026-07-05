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

            print(f"Loading map {self.cfg.map.town}")

            self.world = self.client.load_world_if_different(
                self.cfg.map.town,
                reset_settings=False,
            )

        else:

            print("Reusing existing world")

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

        print("World:", self.world.id)
        print("Map:", self.world.get_map().name)
        print(
            "Vehicles:",
            len(
                self.world.get_actors().filter(
                    "vehicle.*"
                )
            ),
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

    def tick(self):

        return self.world.tick()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):

        if self.actors is not None:
            self.actors.destroy_all()

        self.world.apply_settings(
            self.original_settings
        )

        self.logger.info(
            "World closed."
        )

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