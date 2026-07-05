import random

import carla

from lane_change_rl.config import load_config

class CarlaWorld:

    def __init__(self, cfg):

        self.cfg = cfg

        self.client = carla.Client(
            self.cfg.carla.host,
            self.cfg.carla.port
        )

        self.client.set_timeout(self.cfg.carla.timeout)

        self.world = self.client.load_world(
            self.cfg.map.town
        )

        self.blueprints = self.world.get_blueprint_library()

        self.actors = []

        self.ego = None

        self.tm = self.client.get_trafficmanager(
            self.cfg.traffic.tm_port
        )

        self.tm.set_synchronous_mode(True)

        self._setup_world()

    def _setup_world(self):

        settings = self.world.get_settings()

        settings.synchronous_mode = (
            self.cfg.simulation.synchronous_mode
        )

        settings.fixed_delta_seconds = (
            self.cfg.simulation.fixed_delta_seconds
        )

        self.world.apply_settings(settings)

    def tick(self):

        self.world.tick()

    def destroy(self):

        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()

        self.actors.clear()

        self.ego = None

    def spawn_ego(self):

        bp = self.blueprints.find(
            self.cfg.ego.blueprint
        )

        spawn = random.choice(
            self.world.get_map().get_spawn_points()
        )

        vehicle = self.world.try_spawn_actor(
            bp,
            spawn
        )

        if vehicle is None:
            raise RuntimeError(
                "Cannot spawn ego vehicle."
            )

        vehicle.set_autopilot(
            True,
self.cfg.traffic.tm_port        )

        self.actors.append(vehicle)

        self.ego = vehicle

        return vehicle

    def spawn_traffic(self):

        vehicle_bps = self.blueprints.filter(
            "vehicle.*"
        )

        spawn_points = self.world.get_map().get_spawn_points()

        random.shuffle(spawn_points)

        count = 0

        for spawn in spawn_points:

            if count >= self.cfg.traffic.num_vehicles:
                break

            bp = random.choice(vehicle_bps)

            vehicle = self.world.try_spawn_actor(
                bp,
                spawn
            )

            if vehicle is None:
                continue

            vehicle.set_autopilot(
                True,
                self.cfg.traffic.tm_port
            )

            self.tm.auto_lane_change(vehicle, True)

            self.actors.append(vehicle)

            count += 1

    def get_speed(self):

        vel = self.ego.get_velocity()

        return (
            3.6
            * (
                vel.x ** 2
                + vel.y ** 2
                + vel.z ** 2
            ) ** 0.5
        )

    def get_lane_id(self):

        location = self.ego.get_location()

        waypoint = self.world.get_map().get_waypoint(
            location
        )

        return waypoint.lane_id

    def spectator_follow(self):

        spectator = self.world.get_spectator()

        transform = self.ego.get_transform()

        location = transform.location + carla.Location(
            z=25
        )

        rotation = carla.Rotation(
            pitch=-90
        )

        spectator.set_transform(
            carla.Transform(
                location,
                rotation
            )
        )
