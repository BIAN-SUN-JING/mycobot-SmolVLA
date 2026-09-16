import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "work_pose"
)


def save_snapshot(frame, name):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / name

    print("[Snapshot] Writing image...")

    ok = cv2.imwrite(
        str(path),
        frame,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save image: {path}"
        )

    print("[Snapshot] Image saved.")

    return path


def get_camera_frame(robot):
    if "camera1" not in robot.cameras:
        raise RuntimeError(
            "camera1 not found."
        )

    camera = robot.cameras["camera1"]

    print(
        "[Camera] Capturing frame..."
    )

    frame = camera.async_read(
        timeout_ms=2000
    )

    print(
        "[Camera] Frame received:",
        frame.shape,
    )

    return frame


def get_joint_angles(robot):
    if robot.mc is None:
        raise RuntimeError(
            "Robot controller is unavailable."
        )

    print(
        "[Robot] Reading joint angles..."
    )

    angles = robot.mc.get_angles()

    if angles is None or len(angles) != 6:
        raise RuntimeError(
            f"Invalid joint angles: {angles}"
        )

    print(
        "[Robot] Joint angles received."
    )

    return [
        float(x)
        for x in angles
    ]


def main():
    config = MyCobot280RobotConfig(
        id="mycobot280_pose_setup",
        port="auto",
        baudrate=1_000_000,
        enable_motion=False,
    )

    robot = MyCobot280Robot(
        config
    )

    released = False

    try:
        print()
        print("=" * 60)
        print("myCobot WORK POSE SETUP")
        print("=" * 60)

        print()
        print(
            "Connecting robot and camera..."
        )

        robot.connect(
            calibrate=False
        )

        print()
        print(
            "ROBOT_CONNECTED"
        )

        # ====================================================
        # Initial state
        # ====================================================

        initial_angles = (
            get_joint_angles(robot)
        )

        print()
        print(
            "Initial joint angles:"
        )

        print(
            [
                round(x, 2)
                for x in initial_angles
            ]
        )

        initial_frame = (
            get_camera_frame(robot)
        )

        initial_path = (
            save_snapshot(
                initial_frame,
                "before_unlock.jpg",
            )
        )

        print()
        print(
            "Initial camera image:"
        )

        print(initial_path)

        print()
        print(
            "Open it with:"
        )

        print(
            f'start "" "{initial_path}"'
        )

        # ====================================================
        # Safety
        # ====================================================

        print()
        print("=" * 60)
        print("SAFETY")
        print("=" * 60)

        print()
        print(
            "Hold/support the robot arm "
            "BEFORE unlocking."
        )

        print(
            "The arm may drop when "
            "servos are released."
        )

        print()

        input(
            "Press ENTER when you are "
            "holding the arm safely..."
        )

        print()
        print(
            "Unlocking in:"
        )

        for value in [3, 2, 1]:
            print(value)
            time.sleep(1)

        if robot.mc is None:
            raise RuntimeError(
                "Robot controller unavailable."
            )

        # ====================================================
        # Free drive
        # ====================================================

        print()
        print(
            "Releasing all servos..."
        )

        robot.mc.release_all_servos()

        released = True

        time.sleep(0.5)

        print()
        print("=" * 60)
        print("FREE_DRIVE_ACTIVE")
        print("=" * 60)

        print()
        print(
            "Move the robot gently by hand."
        )

        print()
        print(
            "Commands:"
        )

        print(
            "  ENTER / s : save camera snapshot"
        )

        print(
            "  l         : lock robot and finish"
        )

        print(
            "  q         : abort and lock robot"
        )

        print()
        print(
            "NOTE:"
        )

        print(
            "Joint angles are NOT read while "
            "the robot is unlocked."
        )

        print()

        snapshot_index = 0

        while True:
            command = input(
                "work-pose> "
            ).strip().lower()

            # ================================================
            # Snapshot
            # ================================================

            if command in (
                "",
                "s",
            ):
                print()
                print(
                    "[Snapshot] Starting..."
                )

                try:
                    frame = (
                        get_camera_frame(
                            robot
                        )
                    )

                    snapshot_index += 1

                    filename = (
                        f"preview_"
                        f"{snapshot_index:03d}.jpg"
                    )

                    path = (
                        save_snapshot(
                            frame,
                            filename,
                        )
                    )

                    print()
                    print(
                        "Snapshot saved:"
                    )

                    print(path)

                    print()
                    print(
                        "Open it in another "
                        "PowerShell:"
                    )

                    print(
                        f'start "" "{path}"'
                    )

                except Exception as exc:
                    print()
                    print(
                        "[Snapshot ERROR]"
                    )

                    print(
                        repr(exc)
                    )

                print()

            # ================================================
            # Lock
            # ================================================

            elif command == "l":
                print()
                print(
                    "Lock requested."
                )

                break

            # ================================================
            # Abort
            # ================================================

            elif command == "q":
                print()
                print(
                    "Abort requested."
                )

                break

            else:
                print(
                    "Unknown command."
                )

                print(
                    "Use ENTER/s, l, or q."
                )

        # ====================================================
        # Lock robot
        # ====================================================

        print()
        print(
            "Locking robot..."
        )

        robot.mc.power_on()

        released = False

        time.sleep(2)

        print(
            "Robot locked."
        )

        # ====================================================
        # Read final state AFTER locking
        # ====================================================

        final_angles = (
            get_joint_angles(robot)
        )

        final_frame = (
            get_camera_frame(robot)
        )

        final_path = (
            save_snapshot(
                final_frame,
                "final_work_pose.jpg",
            )
        )

        print()
        print("=" * 60)
        print("FINAL WORK POSE")
        print("=" * 60)

        print()
        print(
            [
                round(x, 2)
                for x in final_angles
            ]
        )

        print()
        print(
            "power:",
            robot.mc.is_power_on(),
        )

        print(
            "servos:",
            robot.mc.is_all_servo_enable(),
        )

        print(
            "error:",
            robot.mc.get_error_information(),
        )

        print()
        print(
            "Final camera image:"
        )

        print(
            final_path
        )

        print()
        print(
            "Open final image with:"
        )

        print(
            f'start "" "{final_path}"'
        )

        print()
        print("=" * 60)
        print("WORK_POSE_SETUP_OK")
        print("=" * 60)

    except KeyboardInterrupt:
        print()
        print()
        print(
            "Keyboard interrupt received."
        )

    finally:
        # ====================================================
        # CRITICAL SAFETY FALLBACK
        # ====================================================

        if robot.mc is not None:
            try:
                if released:
                    print()
                    print(
                        "[Safety] "
                        "Re-locking robot..."
                    )

                robot.mc.power_on()

                time.sleep(1)

            except Exception as exc:
                print()
                print(
                    "[Safety WARNING]"
                )

                print(
                    "Could not re-lock robot:"
                )

                print(
                    repr(exc)
                )

        try:
            robot.disconnect()

        except Exception:
            pass


if __name__ == "__main__":
    main()