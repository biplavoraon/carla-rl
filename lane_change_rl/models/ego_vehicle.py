"""
Data model representing the ego vehicle and all sensors attached to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import carla


@dataclass(slots=True)
class EgoVehicle:
    """
    Container for the ego vehicle and its attached sensors.
    """

    vehicle: carla.Vehicle

    rgb_camera: carla.Sensor | None = None

    collision_sensor: carla.Sensor | None = None

    lane_invasion_sensor: carla.Sensor | None = None

    lidar: carla.Sensor | None = None

    radar: carla.Sensor | None = None

    gnss: carla.Sensor | None = None

    imu: carla.Sensor | None = None

    def destroy(self) -> None:
        """
        Destroy every attached sensor followed by the vehicle.
        """

        sensors = [
            self.rgb_camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.lidar,
            self.radar,
            self.gnss,
            self.imu,
        ]

        for sensor in sensors:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()

        if self.vehicle.is_alive:
            self.vehicle.destroy()