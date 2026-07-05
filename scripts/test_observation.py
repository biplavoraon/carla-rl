from lane_change_rl.env.world import CarlaWorld
from lane_change_rl.env.observation import ObservationExtractor

import carla

with CarlaWorld() as sim:
    ego = sim.actors.spawn_ego()

    sim.tick()

    spectator = sim.world.get_spectator()

    transform = ego.vehicle.get_transform()

    spectator.set_transform(
        carla.Transform(
            transform.transform(
                carla.Location(
                    x=-6.0,   # 6 m behind
                    z=2.5,    # 2.5 m above
                )
            ),
            transform.rotation,
        )
    )

    print("Returned handle:")
    print(ego.vehicle.id)
    print(ego.vehicle.is_alive)

    actors = sim.world.get_actors().filter("vehicle.*")

    print("Vehicles currently in world:")

    for actor in actors:
        print(
            actor.id,
            actor.type_id,
            actor.is_alive,
            actor.get_location(),
        )

    sim.actors.spawn_background_traffic()

    ego.vehicle.set_autopilot(
        True,
        sim.traffic.port,
    )

    observer = ObservationExtractor(sim)


    while True:

        sim.tick()

        transform = ego.vehicle.get_transform()

        spectator.set_transform(
            carla.Transform(
                transform.transform(
                    carla.Location(
                        x=-6.0,
                        z=2.5,
                    )
                ),
                transform.rotation,
            )
        )

        obs = observer.observe()


        print(f"Speed (km/h)        : {obs[0] * 3.6 * 30:.2f}")
        print(f"Lane         : {obs[1]}")
        print(f"Heading      : {obs[2]:.3f}")
        print(f"Front gap    : {obs[3]:.2f}")
        print(f"Front Δspeed : {obs[4]:.2f}")
        print(f"Left front   : {obs[5]:.2f}")
        print(f"Left rear    : {obs[6]:.2f}")
        print(f"Right front  : {obs[7]:.2f}")
        print(f"Right rear   : {obs[8]:.2f}")
        print(f"Speed limit  : {obs[9]:.2f}")
        print(f"Lane offset  : {obs[10]:.3f}")

        print("Ego ID:", ego.vehicle.id)

        for actor in sim.world.get_actors().filter("vehicle.*"):
            if actor.id == ego.vehicle.id:
                print(
                    "Found ego:",
                    actor.get_location(),
                    actor.get_velocity(),
                )
