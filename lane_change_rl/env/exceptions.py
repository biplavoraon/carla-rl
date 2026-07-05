"""
Custom exceptions for the Lane Change RL project.

Using project-specific exceptions instead of generic RuntimeError makes
debugging and error handling much cleaner.
"""


class LaneChangeRLError(Exception):
    """
    Base exception for the project.
    """

    pass


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class ConfigurationError(LaneChangeRLError):
    """
    Raised when configuration loading or validation fails.
    """

    pass


# ---------------------------------------------------------------------
# CARLA Connection
# ---------------------------------------------------------------------


class CarlaConnectionError(LaneChangeRLError):
    """
    Raised when a connection to the CARLA server cannot be established.
    """

    pass


class WorldInitializationError(LaneChangeRLError):
    """
    Raised when the CARLA world cannot be initialized.
    """

    pass


# ---------------------------------------------------------------------
# Actor Management
# ---------------------------------------------------------------------


class SpawnError(LaneChangeRLError):
    """
    Raised when an actor cannot be spawned.
    """

    pass


class ActorNotFoundError(LaneChangeRLError):
    """
    Raised when an expected actor no longer exists.
    """

    pass


class ActorDestroyedError(LaneChangeRLError):
    """
    Raised when an operation is attempted on a destroyed actor.
    """

    pass


# ---------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------


class SensorError(LaneChangeRLError):
    """
    Base class for sensor-related exceptions.
    """

    pass


class CameraInitializationError(SensorError):
    """
    RGB camera failed to initialize.
    """

    pass


class CollisionSensorError(SensorError):
    """
    Collision sensor failed.
    """

    pass


class LaneInvasionSensorError(SensorError):
    """
    Lane invasion sensor failed.
    """

    pass


# ---------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------


class TrafficManagerError(LaneChangeRLError):
    """
    Traffic Manager configuration failed.
    """

    pass


# ---------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------


class MapError(LaneChangeRLError):
    """
    Map-related operation failed.
    """

    pass


class WaypointError(MapError):
    """
    Failed to retrieve a waypoint.
    """

    pass


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------


class EnvironmentError(LaneChangeRLError):
    """
    RL environment error.
    """

    pass


class InvalidActionError(EnvironmentError):
    """
    Agent produced an invalid action.
    """

    pass


class EpisodeTerminatedError(EnvironmentError):
    """
    Operation attempted after episode termination.
    """

    pass