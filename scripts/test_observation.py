import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


def main():
    config = MyCobot280RobotConfig(
        id="mycobot280_main",
        port="auto",
        baudrate=1_000_000,
        enable_motion=False,
    )

    robot = MyCobot280Robot(config)

    print()
    print("=== FEATURES ===")
    print("Observation:")
    print(robot.observation_features)

    print()
    print("Action:")
    print(robot.action_features)

    print()
    print("=== CONNECT ===")

    try:
        robot.connect(
            calibrate=False
        )

        print()
        print(
            "is_connected:",
            robot.is_connected,
        )

        print()
        print("=== OBSERVATION ===")

        obs = robot.get_observation()

        for key, value in obs.items():
            if hasattr(value, "shape"):
                print(
                    f"{key}: "
                    f"shape={value.shape}, "
                    f"dtype={value.dtype}"
                )
            else:
                print(
                    f"{key}: {value}"
                )

        print()
        print("READ_ONLY_TEST_OK")

        print()
        print("=== SAFETY TEST ===")

        fake_action = {
            f"joint_{i}.pos": 0.0
            for i in range(1, 7)
        }

        try:
            robot.send_action(
                fake_action
            )

            raise RuntimeError(
                "SAFETY FAILURE: "
                "send_action unexpectedly succeeded."
            )

        except RuntimeError as exc:
            if "DISABLED" in str(exc):
                print(
                    "MOTION_BLOCK_OK"
                )
            else:
                raise

    finally:
        print()
        print("=== DISCONNECT ===")
        robot.disconnect()


if __name__ == "__main__":
    main()