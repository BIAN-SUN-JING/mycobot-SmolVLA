import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


TARGET_DELTA = 3.0
TOLERANCE = 1.0
OTHER_JOINT_TOLERANCE = 1.0


def main():
    config = MyCobot280RobotConfig(
        id="mycobot280_main",
        port="auto",
        baudrate=1_000_000,
        enable_motion=True,
        motion_speed=20,
        max_delta_deg=3.0,
    )

    robot = MyCobot280Robot(config)

    try:
        robot.connect(
            calibrate=False
        )

        obs = robot.get_observation()

        original = [
            float(
                obs[f"joint_{i}.pos"]
            )
            for i in range(1, 7)
        ]

        print()
        print("Original:")
        print(original)

        target = original.copy()
        target[0] += TARGET_DELTA

        action = {
            f"joint_{i + 1}.pos":
                target[i]
            for i in range(6)
        }

        print()
        print(
            f"Target J1: "
            f"{target[0]:.2f}"
        )

        robot.send_action(action)

        # 等待运动
        time.sleep(3.0)

        obs_after = robot.get_observation()

        after = [
            float(
                obs_after[
                    f"joint_{i}.pos"
                ]
            )
            for i in range(1, 7)
        ]

        print()
        print("After:")
        print(after)

        j1_error = abs(
            after[0] - target[0]
        )

        other_drift = [
            abs(
                after[i] - original[i]
            )
            for i in range(1, 6)
        ]

        print()
        print(
            "J1 target error:",
            round(j1_error, 3),
        )

        print(
            "Other joint drift:",
            [
                round(x, 3)
                for x in other_drift
            ],
        )

        if j1_error > TOLERANCE:
            raise RuntimeError(
                f"J1 did not reach target. "
                f"error={j1_error:.2f} deg"
            )

        if max(other_drift) > OTHER_JOINT_TOLERANCE:
            raise RuntimeError(
                "Unexpected movement detected "
                f"in another joint: "
                f"{other_drift}"
            )

        print()
        print(
            "FORWARD_MOTION_OK"
        )

        # 返回原始姿态
        return_action = {
            f"joint_{i + 1}.pos":
                original[i]
            for i in range(6)
        }

        print()
        print("Returning J1...")

        robot.send_action(
            return_action
        )

        time.sleep(3.0)

        obs_final = robot.get_observation()

        final = [
            float(
                obs_final[
                    f"joint_{i}.pos"
                ]
            )
            for i in range(1, 7)
        ]

        print()
        print("Final:")
        print(final)

        return_error = abs(
            final[0] - original[0]
        )

        print(
            "Return error:",
            round(return_error, 3),
        )

        if return_error > TOLERANCE:
            raise RuntimeError(
                f"J1 failed to return. "
                f"error={return_error:.2f} deg"
            )

        print()
        print(
            "VERIFIED_MOTION_TEST_OK"
        )

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()