import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymycobot import MyCobot280

from hardware.mycobot_robot import (
    MyCobot280Robot,
)


def main():
    port = MyCobot280Robot.find_mycobot_port()

    print(f"[myCobot] Port: {port}")

    mc = MyCobot280(
        port,
        1_000_000,
    )

    try:
        time.sleep(2)

        print()
        print("Initial angles:")
        print(mc.get_angles())

        print()
        print(
            "IMPORTANT: Hold the robot arm "
            "before servos are released."
        )

        print(
            "Free-drive starts in 5 seconds..."
        )

        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)

        print()
        print("Releasing servos...")

        mc.release_all_servos()

        time.sleep(1)

        print()
        print(
            "FREE_DRIVE_ACTIVE"
        )

        print(
            "Move the robot gently by hand."
        )

        # 读取 10 秒
        start = time.time()

        while time.time() - start < 10:
            angles = mc.get_angles()

            print(
                "\rAngles: "
                + str(
                    [
                        round(x, 2)
                        for x in angles
                    ]
                ),
                end="",
            )

            time.sleep(0.2)

        print()
        print()

        print(
            "Locking robot..."
        )

        mc.power_on()

        time.sleep(2)

        print()
        print(
            "power:",
            mc.is_power_on(),
        )

        print(
            "servos:",
            mc.is_all_servo_enable(),
        )

        print(
            "final angles:",
            mc.get_angles(),
        )

        print()
        print(
            "FREE_DRIVE_TEST_OK"
        )

    finally:
        try:
            mc.power_on()
        except Exception:
            pass

        try:
            mc.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()