import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets import LeRobotDataset

from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


# =========================
# Recording configuration
# =========================

FPS = 10
EPISODE_SECONDS = 8.0

TASK = (
    "Move the end effector above the red cube."
)

DATASET_REPO_ID = (
    "local/mycobot_reach_red_cube_smoke_v0"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "mycobot_reach_red_cube_smoke_v0"
)

JOINT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
]


def observation_to_state(obs):
    return np.asarray(
        [
            obs[name]
            for name in JOINT_NAMES
        ],
        dtype=np.float32,
    )


def wait_until(target_time):
    now = time.perf_counter()

    remaining = target_time - now

    if remaining > 0:
        time.sleep(remaining)


def main():
    print()
    print("===================================")
    print("myCobot SmolVLA dataset smoke test")
    print("===================================")

    print()
    print("Task:")
    print(TASK)

    print()
    print("Dataset root:")
    print(DATASET_ROOT)

    # 防止意外覆盖已有测试数据
    if DATASET_ROOT.exists():
        raise FileExistsError(
            f"\nDataset already exists:\n"
            f"{DATASET_ROOT}\n\n"
            f"Delete it manually before "
            f"recording again."
        )

    DATASET_ROOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    robot_config = MyCobot280RobotConfig(
        id="mycobot280_main",
        port="auto",
        baudrate=1_000_000,
        enable_motion=False,
    )

    robot = MyCobot280Robot(
        robot_config
    )

    # 标准 SmolVLA / LeRobot feature names
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },

        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },

        "observation.images.camera1": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=DATASET_ROOT,
        robot_type="mycobot280",
        features=features,
        use_videos=True,
    )

    episode_saved = False

    try:
        print()
        print("Connecting robot...")

        robot.connect(
            calibrate=False
        )

        print()
        print("Initial angles:")

        initial_obs = robot.get_observation()

        print(
            observation_to_state(
                initial_obs
            ).tolist()
        )

        print()
        print(
            "Place the RED CUBE in the "
            "camera workspace."
        )

        print(
            "Set the robot to a safe "
            "starting configuration."
        )

        print()
        print(
            "Hold the robot arm firmly."
        )

        input(
            "Press ENTER when ready..."
        )

        print()
        print("Recording starts in:")

        for value in [3, 2, 1]:
            print(value)
            time.sleep(1)

        if robot.mc is None:
            raise RuntimeError(
                "Robot controller missing."
            )

        print()
        print("Releasing servos...")

        robot.mc.release_all_servos()

        time.sleep(0.3)

        print()
        print("===================================")
        print("RECORDING")
        print("===================================")

        print(
            "Move the end effector "
            "smoothly above the red cube."
        )

        print(
            "Hold the final pose briefly."
        )

        print()

        period = 1.0 / FPS

        start_time = time.perf_counter()
        next_time = start_time

        previous_state = None
        previous_image = None

        frame_count = 0

        while True:
            now = time.perf_counter()

            elapsed = now - start_time

            if elapsed >= EPISODE_SECONDS:
                break

            obs = robot.get_observation()

            state = observation_to_state(
                obs
            )

            image = obs["camera1"]

            # 第一个 observation 没有前一帧，
            # 所以从第二次采样开始创建 transition。
            #
            # previous_state = observation(t)
            # state          = observation(t+1)
            #
            # 因此：
            # action(t) = state(t+1)
            #
            # 这符合我们当前绝对关节目标控制方式。
            if previous_state is not None:
                dataset.add_frame(
                    {
                        "observation.state":
                            previous_state,

                        "action":
                            state.copy(),

                        "observation.images.camera1":
                            previous_image,

                        "task":
                            TASK,
                    }
                )

                frame_count += 1

            previous_state = state.copy()
            previous_image = image.copy()

            next_time += period

            wait_until(
                next_time
            )

        # 最后一帧使用保持动作
        if previous_state is not None:
            dataset.add_frame(
                {
                    "observation.state":
                        previous_state,

                    "action":
                        previous_state.copy(),

                    "observation.images.camera1":
                        previous_image,

                    "task":
                        TASK,
                }
            )

            frame_count += 1

        print()
        print("Recording finished.")

        # 第一时间重新锁住机械臂
        print()
        print("Locking robot...")

        robot.mc.power_on()

        time.sleep(2)

        print(
            "power:",
            robot.mc.is_power_on(),
        )

        print(
            "servos:",
            robot.mc.is_all_servo_enable(),
        )

        print()
        print(
            f"Frames recorded: "
            f"{frame_count}"
        )

        actual_duration = (
            time.perf_counter()
            - start_time
        )

        effective_fps = (
            frame_count
            / actual_duration
        )

        print(
            f"Effective FPS: "
            f"{effective_fps:.2f}"
        )

        print()
        print(
            "Saving episode..."
        )

        dataset.save_episode()

        episode_saved = True

        print(
            "EPISODE_SAVE_OK"
        )

    except KeyboardInterrupt:
        print()
        print(
            "Recording interrupted."
        )

        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()

        raise

    finally:
        # 无论发生什么，都尽量重新锁住机器人
        if robot.mc is not None:
            try:
                robot.mc.power_on()
            except Exception:
                pass

        try:
            robot.disconnect()
        except Exception:
            pass

        if (
            not episode_saved
            and dataset.has_pending_frames()
        ):
            dataset.clear_episode_buffer()

        print()
        print(
            "Finalizing dataset..."
        )

        dataset.finalize()

        print(
            "DATASET_FINALIZE_OK"
        )

    print()
    print("===================================")
    print("SMOKE DATASET RECORDING SUCCESS")
    print("===================================")


if __name__ == "__main__":
    main()