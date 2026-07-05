"""
Traffic Manager wrapper.

This module encapsulates all interaction with CARLA's Traffic Manager.
"""

from __future__ import annotations

import carla

from configs.loader import TrafficConfig


class TrafficManager:

    """
    Wrapper around CARLA TrafficManager.
    """

    def __init__(
        self,
        client: carla.Client,
        config: TrafficConfig,
    ):

        self._config = config

        self._tm = client.get_trafficmanager(
            config.tm_port
        )

        self._initialize()

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def _initialize(self):

        self._tm.set_synchronous_mode(True)

        self._tm.set_random_device_seed(
            self._config.seed
        )

        self._tm.set_hybrid_physics_mode(
            self._config.hybrid_physics
        )

        self._tm.global_percentage_speed_difference(
            self._config.global_speed_difference
        )

    # ---------------------------------------------------------
    # Vehicle Registration
    # ---------------------------------------------------------

    def register_background_vehicle(
        self,
        vehicle: carla.Vehicle,
    ):
        vehicle.set_autopilot(
            True,
            self.port,
        )

        self._tm.auto_lane_change(
            vehicle,
            self._config.auto_lane_change,
        )


    def register_ego_vehicle(
        self,
        vehicle: carla.Vehicle,
    ):
        """
        Reserved for future use.

        The RL agent will control the ego vehicle,
        so we intentionally do NOT enable autopilot.
        """
        pass

    # ---------------------------------------------------------
    # Per Vehicle Controls
    # ---------------------------------------------------------

    def enable_lane_change(
        self,
        vehicle: carla.Vehicle,
    ):

        self._tm.auto_lane_change(
            vehicle,
            True,
        )

    def disable_lane_change(
        self,
        vehicle: carla.Vehicle,
    ):

        self._tm.auto_lane_change(
            vehicle,
            False,
        )

    def ignore_lights(
        self,
        vehicle: carla.Vehicle,
        percent: float = 100,
    ):

        self._tm.ignore_lights_percentage(
            vehicle,
            percent,
        )

    def ignore_signs(
        self,
        vehicle: carla.Vehicle,
        percent: float = 100,
    ):

        self._tm.ignore_signs_percentage(
            vehicle,
            percent,
        )

    def ignore_walkers(
        self,
        vehicle: carla.Vehicle,
        percent: float = 100,
    ):

        self._tm.ignore_walkers_percentage(
            vehicle,
            percent,
        )

    def set_speed_difference(
        self,
        vehicle: carla.Vehicle,
        percent: float,
    ):

        self._tm.vehicle_percentage_speed_difference(
            vehicle,
            percent,
        )

    def set_follow_distance(
        self,
        vehicle: carla.Vehicle,
        meters: float,
    ):

        self._tm.distance_to_leading_vehicle(
            vehicle,
            meters,
        )

    def force_lane_change(
        self,
        vehicle: carla.Vehicle,
        to_right: bool,
    ):

        self._tm.force_lane_change(
            vehicle,
            to_right,
        )

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def port(self):

        return self._config.tm_port

    @property
    def manager(self):

        return self._tm