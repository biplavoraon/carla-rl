from lane_change_rl.env.world import CarlaWorld


def main():

    with CarlaWorld() as sim:

        sim.actors.spawn_ego()

        sim.actors.spawn_background_traffic()

        while True:

            sim.tick()


if __name__ == "__main__":
    main()