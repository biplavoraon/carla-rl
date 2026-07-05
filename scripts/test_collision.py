from lane_change_rl.env.world import CarlaWorld
from lane_change_rl.sensors import CollisionSensor


def main():

    with CarlaWorld() as sim:

        ego = sim.actors.spawn_ego()

        sensor = CollisionSensor(
            sim.world,
            sim.blueprints,
        )

        sensor.spawn(ego.vehicle)

        ego.vehicle.set_autopilot(
            True,
            sim.traffic.port,
        )


        print("Drive into another vehicle...")
        print(ego)

        while True:

            sim.tick()

            if sensor.has_collision:

                print("Collision!")

                print(
                    sensor.last_event.other_actor.type_id
                )

                break


if __name__ == "__main__":
    main()