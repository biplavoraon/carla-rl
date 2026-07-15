from dataclasses import dataclass

import carla

import numpy as np


MAX_DISTANCE = 80.0
MAX_SPEED = 30.0  # m/s (~108 km/h)


@dataclass(slots=True)
class VehicleInfo:
    vehicle: carla.Vehicle | None

    distance: float

    relative_speed: float


@dataclass(slots=True)
class LaneInfo:
    front: VehicleInfo

    rear: VehicleInfo

    exists: bool

class ObservationExtractor:

    def __init__(self, world):

        self.world = world

        self.map = world.map

        self.actors = world.actors


    def observe(self):

        ego = self.actors.ego_vehicle

        # print(
        #     "ego:",
        #     ego,
        #     "alive:",
        #     None if ego is None else ego.is_alive,
        # )

        if ego is None:
            raise RuntimeError("Ego vehicle not registered")

        if not ego.is_alive:
            raise RuntimeError("Ego vehicle is not alive")

        wp = self.map.get_waypoint(
            ego.get_location()
        )

        current = self._current_lane(
            ego,
            wp,
        )

        left = self._adjacent_lane(
            ego,
            wp.get_left_lane(),
        )

        right = self._adjacent_lane(
            ego,
            wp.get_right_lane(),
        )

        return self._build_vector(
            ego,
            wp,
            current,
            left,
            right,
        )

    def _collect_nearby_vehicles(
        self,
        ego: carla.Vehicle,
    ):

        ego_location = ego.get_location()

        nearby = []

        for vehicle in self.actors.traffic:

            if vehicle.id == ego.id:
                continue

            distance = ego_location.distance(
                vehicle.get_location()
            )

            # Ignore distant vehicles
            if distance > 80.0:
                continue

            nearby.append(vehicle)

        return nearby

    def _projection(
        self,
        ego: carla.Vehicle,
        other: carla.Vehicle,
    ):

        ego_tf = ego.get_transform()

        forward = ego_tf.get_forward_vector()

        delta = (
            other.get_location()
            - ego_tf.location
        )

        return (
            delta.x * forward.x +
            delta.y * forward.y +
            delta.z * forward.z
        )


    def _current_lane(
        self,
        ego,
        waypoint,
    ):

        front = VehicleInfo(
            None,
            float("inf"),
            0.0,
        )

        rear = VehicleInfo(
            None,
            float("inf"),
            0.0,
        )

        ego_speed = self._speed(ego)

        for vehicle in self._collect_nearby_vehicles(ego):

            wp = self.map.get_waypoint(
                vehicle.get_location()
            )

            if wp.road_id != waypoint.road_id:
                continue

            if wp.lane_id != waypoint.lane_id:
                continue

            distance = ego.get_location().distance(
                vehicle.get_location()
            )

            relative_speed = (
                self._speed(vehicle)
                - ego_speed
            )

            projection = self._projection(
                ego,
                vehicle,
            )

            if projection > 0:

                if distance < front.distance:

                    front = VehicleInfo(
                        vehicle,
                        distance,
                        relative_speed,
                    )

            else:

                if distance < rear.distance:

                    rear = VehicleInfo(
                        vehicle,
                        distance,
                        relative_speed,
                    )

        return LaneInfo(
            front=front,
            rear=rear,
            exists=True,
        )


    def _adjacent_lane(
        self,
        ego: carla.Vehicle,
        lane: carla.Waypoint | None,
    ) -> LaneInfo:

        if lane is None:

            empty = VehicleInfo(
                vehicle=None,
                distance=float("inf"),
                relative_speed=0.0,
            )

            return LaneInfo(
                front=empty,
                rear=empty,
                exists=False,
            )

        front = VehicleInfo(
            vehicle=None,
            distance=float("inf"),
            relative_speed=0.0,
        )

        rear = VehicleInfo(
            vehicle=None,
            distance=float("inf"),
            relative_speed=0.0,
        )

        ego_speed = self._speed(ego)

        for vehicle in self._collect_nearby_vehicles(ego):

            wp = self.map.get_waypoint(
                vehicle.get_location()
            )

            if wp.road_id != lane.road_id:
                continue

            if wp.lane_id != lane.lane_id:
                continue

            distance = ego.get_location().distance(
                vehicle.get_location()
            )

            relative_speed = (
                self._speed(vehicle)
                - ego_speed
            )

            projection = self._projection(
                ego,
                vehicle,
            )

            info = VehicleInfo(
                vehicle=vehicle,
                distance=distance,
                relative_speed=relative_speed,
            )

            if projection > 0:

                if distance < front.distance:
                    front = info

            else:

                if distance < rear.distance:
                    rear = info

        return LaneInfo(
            front=front,
            rear=rear,
            exists=True,
        )



    def _build_vector(
        self,
        ego: carla.Vehicle,
        waypoint: carla.Waypoint,
        current: LaneInfo,
        left: LaneInfo,
        right: LaneInfo,
    ) -> np.ndarray:

        speed = self._speed(ego)

        heading = self._heading_error(
            ego,
            waypoint,
        )

        lane_offset = (
            ego.get_location().distance(
                waypoint.transform.location
            )
            / max(waypoint.lane_width, 0.1)
        )

        MAX_LANE_ID = 5.0

        target_lane = np.clip(
            self.world.target_lane_id / MAX_LANE_ID,
            -1.0,
            1.0,
        )

        return np.array(
            [
                speed / MAX_SPEED,

                np.clip(
                    waypoint.lane_id / MAX_LANE_ID,
                    -1.0,
                    1.0,
                ),

                np.clip(
                    heading,
                    -1.0,
                    1.0,
                ),

                min(current.front.distance, MAX_DISTANCE)
                / MAX_DISTANCE,

                np.clip(
                    current.front.relative_speed / MAX_SPEED,
                    -1.0,
                    1.0,
                ),

                min(left.front.distance, MAX_DISTANCE)
                / MAX_DISTANCE,

                min(left.rear.distance, MAX_DISTANCE)
                / MAX_DISTANCE,

                min(right.front.distance, MAX_DISTANCE)
                / MAX_DISTANCE,

                min(right.rear.distance, MAX_DISTANCE)
                / MAX_DISTANCE,

                ego.get_speed_limit() / 120.0,

                np.clip(
                    lane_offset,
                    0.0,
                    1.0,
                ),

                target_lane,
            ],
            dtype=np.float32,
        )


    def _heading_error(
        self,
        ego: carla.Vehicle,
        waypoint: carla.Waypoint,
    ) -> float:

        ego_yaw = ego.get_transform().rotation.yaw
        lane_yaw = waypoint.transform.rotation.yaw

        diff = ego_yaw - lane_yaw

        while diff > 180:
            diff -= 360

        while diff < -180:
            diff += 360

        return diff / 180.0


    @staticmethod
    def _speed(
        vehicle: carla.Vehicle,
    ) -> float:

        velocity = vehicle.get_velocity()

        return (
            velocity.x**2 +
            velocity.y**2 +
            velocity.z**2
        ) ** 0.5