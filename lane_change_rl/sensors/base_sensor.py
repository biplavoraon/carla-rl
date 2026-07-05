"""
Base class for all CARLA sensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import carla


class BaseSensor(ABC):
    """
    Base wrapper around a CARLA sensor actor.
    """

    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
    ) -> None:

        self.world = world
        self.blueprints = blueprint_library

        self.sensor: carla.Sensor | None = None

    @property
    def is_alive(self) -> bool:

        return (
            self.sensor is not None
            and self.sensor.is_alive
        )

    @abstractmethod
    def spawn(
        self,
        vehicle: carla.Vehicle,
    ) -> None:
        """
        Spawn and attach the sensor.
        """
        ...

    @abstractmethod
    def _callback(
        self,
        data,
    ) -> None:
        """
        Callback executed whenever new data arrives.
        """
        ...

    def destroy(self) -> None:
        """
        Stop listening and destroy the sensor.
        """

        if self.sensor is None:
            return

        if self.sensor.is_listening:
            self.sensor.stop()

        if self.sensor.is_alive:
            self.sensor.destroy()

        self.sensor = None