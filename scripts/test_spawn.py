from lane_change_rl.env.world import CarlaWorld

with CarlaWorld() as sim:

    ego = sim.actors.spawn_ego()

    sim.tick()

    print("Vehicle:", ego.vehicle)
    print("Alive:", ego.vehicle.is_alive)
    print("Location:", ego.vehicle.get_location())

    print(
        "Vehicles:",
        len(sim.world.get_actors().filter("vehicle.*"))
    )

    sim.tick()

    print("Alive after tick:", ego.vehicle.is_alive)