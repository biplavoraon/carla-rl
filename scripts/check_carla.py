import carla

client=carla.Client("localhost",2000)

client.set_timeout(10)

world=client.get_world()

print(world.get_map().name)

print(len(world.get_actors()))
