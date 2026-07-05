"""
Collision sensor wrapper.
"""

from __future__ import annotations

import threading
from collections import deque

import carla

from lane_change_rl.sensors.base_sensor import BaseSensor


class CollisionSensor(BaseSensor):

    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        history_size: int = 100,
    ):
        super().__init__(world, blueprint_library)

        self._lock = threading.RLock()

        self._history = deque(maxlen=history_size)

        self._last_event: carla.CollisionEvent | None = None

        self._has_collision = False

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def has_collision(self) -> bool:
        with self._lock:
            return self._has_collision

    @property
    def last_event(self) -> carla.CollisionEvent | None:
        with self._lock:
            return self._last_event

    @property
    def history(self):
        with self._lock:
            return tuple(self._history)

    # ---------------------------------------------------------
    # Spawn
    # ---------------------------------------------------------

    def spawn(
        self,
        vehicle: carla.Vehicle,
    ) -> None:

        if self.sensor is not None:
            raise RuntimeError(
                "Collision sensor already spawned."
            )

        bp = self.blueprints.find(
            "sensor.other.collision"
        )

        transform = carla.Transform()

        self.sensor = self.world.spawn_actor(
            bp,
            transform,
            attach_to=vehicle,
        )

        try:

            self.sensor.listen(
                self._callback
            )

        except Exception:

            self.sensor.destroy()

            self.sensor = None

            raise

    # ---------------------------------------------------------
    # Callback
    # ---------------------------------------------------------

    def _callback(
        self,
        event: carla.CollisionEvent,
    ) -> None:

        with self._lock:

            self._has_collision = True

            self._last_event = event

            self._history.append(event)

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def reset(self) -> None:

        with self._lock:

            self._has_collision = False

            self._last_event = None

            self._history.clear()

    def stop(self) -> None:

        if (
            self.sensor is not None
            and self.sensor.is_listening
        ):
            self.sensor.stop()