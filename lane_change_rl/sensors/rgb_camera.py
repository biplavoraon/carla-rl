"""
RGB camera wrapper.
"""

from __future__ import annotations

from queue import Empty, Full, Queue
import threading

import carla
import cv2
import numpy as np

from lane_change_rl.config.loader import CameraConfig
from lane_change_rl.sensors.base_sensor import BaseSensor


class RGBCamera(BaseSensor):

    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        config: CameraConfig,
    ):
        super().__init__(world, blueprint_library)

        self.cfg = config

        self.width = config.width
        self.height = config.height

        self._raw_image: carla.Image | None = None

        self._frame: np.ndarray | None = None

        self._frame_id = -1

        self._timestamp = 0.0

        self._transform: carla.Transform | None = None

        self._queue: Queue[np.ndarray] = Queue(maxsize=1)

        self._lock = threading.RLock()

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._frame is not None


    @property
    def frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id


    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._timestamp


    @property
    def transform(self) -> carla.Transform | None:
        with self._lock:
            return self._transform


    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    # ---------------------------------------------------------
    # Spawn
    # ---------------------------------------------------------

    def spawn(
        self,
        vehicle: carla.Vehicle,
    ):

        if self.sensor is not None:
            raise RuntimeError(
                "RGB camera already exists."
            )

        bp = self.blueprints.find(
            "sensor.camera.rgb"
        )

        bp.set_attribute(
            "image_size_x",
            str(self.width),
        )

        bp.set_attribute(
            "image_size_y",
            str(self.height),
        )

        bp.set_attribute(
            "fov",
            str(self.cfg.fov),
        )

        bp.set_attribute(
            "sensor_tick",
            str(self.cfg.sensor_tick),
        )

        bp.set_attribute(
            "gamma",
            "2.2",
        )

        transform = carla.Transform(
            carla.Location(
                x=self.cfg.x,
                y=self.cfg.y,
                z=self.cfg.z,
            )
        )

        self.sensor = self.world.spawn_actor(
            bp,
            transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
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
        image: carla.Image,
    ):

        self._raw_image = image

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8,
        )

        array = array.reshape(
            (
                image.height,
                image.width,
                4,
            )
        )

        rgb = array[:, :, :3]
        rgb = rgb[:, :, ::-1].copy()

        with self._lock:

            self._frame = rgb

            self._frame_id = image.frame

            self._timestamp = image.timestamp

            self._transform = image.transform

        try:

            if self._queue.full():
                self._queue.get_nowait()

            self._queue.put_nowait(rgb)

        except (Full, Empty):
            pass

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def get(
        self,
        timeout: float | None = None,
    ) -> np.ndarray | None:

        try:
            return self._queue.get(timeout=timeout)

        except Empty:
            return None

    def stop(self) -> None:

        if (
            self.sensor is not None
            and self.sensor.is_listening
        ):
            self.sensor.stop()


    def save(
        self,
        filename: str,
    ):

        if self._raw_image is not None:

            self._raw_image.save_to_disk(
                filename
            )