from lane_change_rl.env.world import CarlaWorld
from lane_change_rl.sensors import RGBCamera

import cv2


def main():

    with CarlaWorld() as sim:

        ego = sim.actors.spawn_ego()

        camera = RGBCamera(
            sim.world,
            sim.blueprints,
            sim.cfg.camera,
        )

        camera.spawn(
            ego.vehicle
        )

        while True:

            sim.tick()

            frame = camera.frame

            if frame is None:
                continue

            cv2.imshow(
                "RGB",
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            if cv2.waitKey(1) == 27:
                break

        camera.destroy()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()