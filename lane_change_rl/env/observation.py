"""
Observation extraction for the Lane Change RL environment.

This module converts the CARLA world into a fixed-length observation vector
for the reinforcement learning agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import carla
import numpy as np

from lane_change_rl.env.utils import vehicle_speed


MAX_DISTANCE = 100.0


@dataclass(slots=True)
class NearbyVehicle:

    vehicle: carla.Vehicle | None

    distance: float

    relative_speed: float


class ObservationExtractor:

    def __init__(self, world):

        self.world = world

        self.map = world.map

        self.actor_manager = world.actors

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def observe(self) -> np.ndarray:

        ego = self.actor_manager.ego_vehicle

        if ego is None:
            raise RuntimeError("No ego vehicle exists.")

        waypoint = self.map.get_waypoint(
            ego.get_location()
        )

        front = self._front_vehicle(
            ego,
            waypoint
        )

        left_front, left_rear = self._lane_neighbours(
            ego,
            waypoint.get_left_lane()
        )

        right_front, right_rear = self._lane_neighbours(
            ego,
            waypoint.get_right_lane()
        )

        observation = np.array(
            [
                self._normalize_speed(
                    vehicle_speed(ego)
                ),

                waypoint.lane_id,

                self._heading_error(
                    ego,
                    waypoint
                ),

                self._normalize_distance(
                    front.distance
                ),

                self._normalize_relative_speed(
                    front.relative_speed
                ),

                self._normalize_distance(
                    left_front.distance
                ),

                self._normalize_distance(
                    left_rear.distance
                ),

                self._normalize_distance(
                    right_front.distance
                ),

                self._normalize_distance(
                    right_rear.distance
                ),

                ego.get_speed_limit() / 120.0,

                self._lane_offset(
                    ego,
                    waypoint
                ),
            ],
            dtype=np.float32,
        )

        return observation

    # --------------------------------------------------------
    # Front vehicle
    # --------------------------------------------------------

    def _front_vehicle(
        self,
        ego,
        waypoint,
    ) -> NearbyVehicle:

        best = NearbyVehicle(
            None,
            MAX_DISTANCE,
            0.0,
        )

        ego_loc = ego.get_location()

        ego_speed = vehicle_speed(ego)

        for vehicle in self.actor_manager.traffic:

            other_wp = self.map.get_waypoint(
                vehicle.get_location()
            )

            if other_wp.road_id != waypoint.road_id:
                continue

            if other_wp.lane_id != waypoint.lane_id:
                continue

            distance = ego_loc.distance(
                vehicle.get_location()
            )

            if distance > best.distance:
                continue

            best = NearbyVehicle(
                vehicle,
                distance,
                vehicle_speed(vehicle) - ego_speed,
            )

        return best

    # --------------------------------------------------------
    # Adjacent lane
    # --------------------------------------------------------

    def _lane_neighbours(
        self,
        ego,
        lane,
    ):

        if lane is None:

            empty = NearbyVehicle(
                None,
                MAX_DISTANCE,
                0,
            )

            return empty, empty

        ego_loc = ego.get_location()

        front = NearbyVehicle(
            None,
            MAX_DISTANCE,
            0,
        )

        rear = NearbyVehicle(
            None,
            MAX_DISTANCE,
            0,
        )

        ego_speed = vehicle_speed(ego)

        for vehicle in self.actor_manager.traffic:

            wp = self.map.get_waypoint(
                vehicle.get_location()
            )

            if wp.road_id != lane.road_id:
                continue

            if wp.lane_id != lane.lane_id:
                continue

            dist = ego_loc.distance(
                vehicle.get_location()
            )

            if vehicle.get_location().x > ego_loc.x:

                if dist < front.distance:

                    front = NearbyVehicle(
                        vehicle,
                        dist,
                        vehicle_speed(vehicle) - ego_speed,
                    )

            else:

                if dist < rear.distance:

                    rear = NearbyVehicle(
                        vehicle,
                        dist,
                        vehicle_speed(vehicle) - ego_speed,
                    )

        return front, rear

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    def _normalize_distance(
        self,
        distance,
    ):

        return min(
            distance,
            MAX_DISTANCE,
        ) / MAX_DISTANCE

    def _normalize_speed(
        self,
        speed,
    ):

        return speed / 120.0

    def _normalize_relative_speed(
        self,
        speed,
    ):

        return np.clip(
            speed / 50.0,
            -1,
            1,
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    def _heading_error(
        self,
        ego,
        waypoint,
    ):

        yaw = ego.get_transform().rotation.yaw

        lane = waypoint.transform.rotation.yaw

        diff = yaw - lane

        return diff / 180.0

    def _lane_offset(
        self,
        ego,
        waypoint,
    ):

        return (
            ego.get_location().distance(
                waypoint.transform.location
            )
            / 4.0
        )