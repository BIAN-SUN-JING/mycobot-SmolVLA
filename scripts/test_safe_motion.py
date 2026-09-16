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


def main():
    config = MyCobot280RobotConfig(
        id="mycobot280_main",
        port="auto",
        baudrate=1_000_000,

        # 本测试显式允许运动
        enable_motion=True,

        motion_speed=5,
        max_delta_deg=1.0,
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
        print(
            "Original:",
            original,
        )

        # 只尝试让 J1 +2°
        target = original.copy()
        target[0] += 1.0

        action = {
            f"joint_{i + 1}.pos":
                target[i]
            for i in range(6)
        }

        print()
        print(
            "Requesting J1 +1 deg..."
        )

        robot.send_action(action)

        time.sleep(3)

        moved = robot.get_observation()

        print()
        print(
            "After movement:",
            [
                moved[
                    f"joint_{i}.pos"
                ]
                for i in range(1, 7)
            ],
        )

        # 回原位
        print()
        print(
            "Returning to original pose..."
        )

        return_action = {
            f"joint_{i + 1}.pos":
                original[i]
            for i in range(6)
        }

        robot.send_action(
            return_action
        )

        time.sleep(3)

        final_obs = robot.get_observation()

        print()
        print(
            "Final:",
            [
                final_obs[
                    f"joint_{i}.pos"
                ]
                for i in range(1, 7)
            ],
        )

        print()
        print(
            "SAFE_MOTION_TEST_OK"
        )

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()