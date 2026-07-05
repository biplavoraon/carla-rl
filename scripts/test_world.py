import time

from env.world import CarlaWorld


def main():

    world = CarlaWorld()

    try:

        world.spawn_ego()

        world.spawn_traffic()

        while True:

            world.tick()

            world.spectator_follow()

            print(
                f"Speed : {world.get_speed():6.2f} km/h"
            )

            print(
                f"Lane  : {world.get_lane_id()}"
            )

            time.sleep(0.05)

    finally:

        world.destroy()


if __name__ == "__main__":
    main()
