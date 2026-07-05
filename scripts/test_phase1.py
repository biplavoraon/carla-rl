from lane_change_rl.env.world import CarlaWorld
from lane_change_rl.env.utils import follow_vehicle


def main():

    with CarlaWorld() as sim:

        ego = sim.actors.spawn_ego()

        print(ego.vehicle.type_id)

        print(ego.vehicle.get_location())

        count = sim.actors.spawn_background_traffic()

        print(count)

        while True:

            follow_vehicle(
                sim.world,
                sim.actors.ego_vehicle,
            )
            
            sim.tick()


if __name__ == "__main__":
    main()