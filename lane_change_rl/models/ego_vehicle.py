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
        Safe to call multiple times.
        """

        sensors = [
            "rgb_camera",
            "collision_sensor",
            "lane_invasion_sensor",
            "lidar",
            "radar",
            "gnss",
            "imu",
        ]

        for name in sensors:

            sensor = getattr(self, name)

            if sensor is None:
                continue

            try:

                sensor.stop()

            except Exception:
                pass

            try:

                if sensor.is_alive:
                    sensor.destroy()

            except RuntimeError:
                pass

            setattr(self, name, None)

        if self.vehicle is not None:

            try:

                if self.vehicle.is_alive:
                    self.vehicle.destroy()

            except RuntimeError:
                pass

            self.vehicle = None