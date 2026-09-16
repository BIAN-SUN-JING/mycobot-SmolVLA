import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame
from lerobot.utils.feature_utils import hw_to_dataset_features

from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


MODEL_ID = "lerobot/smolvla_base"

TASK = "Move the end effector toward the red object."

STATS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "smolvla_stats"
    / "legacy_smolvla_normalization_stats.json"
)

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "j1_visual_calibration"
    / "j1_visual_sign.json"
)


SO100_STATE_MEAN_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.mean"
)

SO100_STATE_STD_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.std"
)


JOINT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
]


J1_MIN = -168.0
J1_MAX = 168.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute J1 motion.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--typical-step-deg",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--deadband",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--center-tolerance-px",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--max-total-travel-deg",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--wait",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_stats():
    data = load_json(
        STATS_FILE
    )

    stats = data["stats"]

    state_mean = torch.tensor(
        stats[
            SO100_STATE_MEAN_KEY
        ]["values"],
        dtype=torch.float32,
    )

    state_std = torch.tensor(
        stats[
            SO100_STATE_STD_KEY
        ]["values"],
        dtype=torch.float32,
    )

    return (
        state_mean,
        state_std,
    )


def load_calibration():
    data = load_json(
        CALIBRATION_FILE
    )

    return {
        "mapping_sign":
            int(
                data[
                    "recommended_mapping_sign"
                ]
            ),

        "neutral":
            float(
                data[
                    "smolvla_dim1_neutral"
                ]
            ),

        "typical_amplitude":
            float(
                data[
                    "smolvla_dim1_typical_amplitude"
                ]
            ),

        "dx_per_deg":
            float(
                data[
                    "dx_per_j1_degree"
                ]
            ),
    }


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_state(robot):
    obs = robot.get_observation()

    state = np.asarray(
        [
            float(obs[name])
            for name in JOINT_NAMES
        ],
        dtype=np.float32,
    )

    return obs, state


def make_red_mask(hsv):
    lower1 = np.array(
        [0, 80, 60],
        dtype=np.uint8,
    )

    upper1 = np.array(
        [12, 255, 255],
        dtype=np.uint8,
    )

    lower2 = np.array(
        [168, 80, 60],
        dtype=np.uint8,
    )

    upper2 = np.array(
        [180, 255, 255],
        dtype=np.uint8,
    )

    mask1 = cv2.inRange(
        hsv,
        lower1,
        upper1,
    )

    mask2 = cv2.inRange(
        hsv,
        lower2,
        upper2,
    )

    return cv2.bitwise_or(
        mask1,
        mask2,
    )


def detect_red(frame):
    candidates = []

    for mode in (
        "RGB",
        "BGR",
    ):
        if mode == "RGB":
            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2HSV,
            )
        else:
            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV,
            )

        mask = make_red_mask(
            hsv
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < 100:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        candidates.append(
            (
                area,
                float(cx),
                float(cy),
            )
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: x[0],
    )


def build_features(robot):
    action_features = (
        hw_to_dataset_features(
            robot.action_features,
            "action",
        )
    )

    observation_features = (
        hw_to_dataset_features(
            robot.observation_features,
            "observation",
        )
    )

    return {
        **action_features,
        **observation_features,
    }


def infer_dim1(
    policy,
    preprocessor,
    observation,
    dataset_features,
    state_mean,
    state_std,
    device,
    seed,
):
    frame = build_inference_frame(
        observation=observation,
        ds_features=dataset_features,
        device=device,
        task=TASK,
        robot_type="mycobot280",
    )

    batch = preprocessor(
        frame
    )

    raw_state = (
        batch[
            "observation.state"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    mean = state_mean.to(
        device
    )

    std = state_std.to(
        device
    )

    normalized_state = (
        raw_state
        - mean
    ) / std

    batch[
        "observation.state"
    ] = normalized_state

    if hasattr(
        policy,
        "reset",
    ):
        policy.reset()

    set_seed(
        seed
    )

    with torch.no_grad():
        action = policy.select_action(
            batch
        )

    action = (
        action
        .detach()
        .float()
        .cpu()
        .flatten()
    )

    return (
        float(action[0]),
        action.tolist(),
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    state_mean, state_std = (
        load_stats()
    )

    calibration = (
        load_calibration()
    )

    mapping_sign = (
        calibration[
            "mapping_sign"
        ]
    )

    neutral = (
        calibration[
            "neutral"
        ]
    )

    amplitude = abs(
        calibration[
            "typical_amplitude"
        ]
    )

    gain = (
        args.typical_step_deg
        / max(
            amplitude,
            1e-6,
        )
    )

    print()
    print("=" * 70)
    print(
        "SmolVLA -> myCobot J1 CLOSED LOOP"
    )
    print("=" * 70)

    print(
        "Mode:",
        (
            "EXECUTE"
            if args.execute
            else "DRY RUN"
        ),
    )

    print(
        "mapping_sign:",
        mapping_sign,
    )

    print(
        "neutral:",
        round(
            neutral,
            6,
        ),
    )

    print(
        "gain:",
        round(
            gain,
            3,
        ),
        "deg / normalized-unit",
    )

    print(
        "deadband:",
        args.deadband,
    )

    print(
        "max step:",
        args.max_step_deg,
        "deg",
    )

    print(
        "max total travel:",
        args.max_total_travel_deg,
        "deg",
    )

    print()
    print(
        "Only J1 may move."
    )

    print(
        "J2-J6 remain fixed."
    )

    print()

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            MODEL_ID
        )
        .to(device)
        .eval()
    )

    preprocessor, _ = (
        make_pre_post_processors(
            policy.config,
            MODEL_ID,
            preprocessor_overrides={
                "device_processor": {
                    "device": str(device),
                }
            },
        )
    )

    config = (
        MyCobot280RobotConfig(
            id="smolvla_j1_closed_loop",
            port="auto",
            baudrate=1_000_000,

            # We do not use robot.send_action().
            enable_motion=False,
        )
    )

    robot = MyCobot280Robot(
        config
    )

    try:
        robot.connect(
            calibrate=False
        )

        dataset_features = (
            build_features(
                robot
            )
        )

        observation, state = (
            get_state(
                robot
            )
        )

        start_j1 = float(
            state[0]
        )

        min_allowed = max(
            J1_MIN,
            start_j1
            - args.max_total_travel_deg,
        )

        max_allowed = min(
            J1_MAX,
            start_j1
            + args.max_total_travel_deg,
        )

        print()
        print(
            "Initial state:",
            [
                round(
                    float(x),
                    3,
                )
                for x in state
            ],
        )

        print(
            f"Allowed J1 envelope: "
            f"[{min_allowed:.2f}, "
            f"{max_allowed:.2f}]"
        )

        previous_pixel_error = None
        worsening_count = 0

        for step_index in range(
            args.max_steps
        ):
            print()
            print(
                "=" * 70
            )

            print(
                f"STEP "
                f"{step_index + 1}/"
                f"{args.max_steps}"
            )

            print(
                "=" * 70
            )

            observation, state = (
                get_state(
                    robot
                )
            )

            frame = observation[
                "camera1"
            ]

            red = detect_red(
                frame
            )

            if red is None:
                print(
                    "STOP: red object not detected."
                )
                break

            area, cx, cy = red

            image_width = (
                frame.shape[1]
            )

            image_center = (
                image_width / 2.0
            )

            pixel_error = (
                cx - image_center
            )

            print(
                f"Red centroid: "
                f"x={cx:.2f}, "
                f"y={cy:.2f}"
            )

            print(
                f"Image center: "
                f"{image_center:.2f}"
            )

            print(
                f"Pixel error: "
                f"{pixel_error:+.2f}px"
            )

            print(
                f"Current J1: "
                f"{state[0]:.3f} deg"
            )

            if (
                abs(pixel_error)
                <= args.center_tolerance_px
            ):
                print()
                print(
                    "TARGET_CENTERED_OK"
                )
                break

            dim1, full_action = (
                infer_dim1(
                    policy=policy,
                    preprocessor=preprocessor,
                    observation=observation,
                    dataset_features=
                        dataset_features,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    seed=args.seed,
                )
            )

            signal = (
                dim1
                - neutral
            )

            print()
            print(
                f"SmolVLA dim1: "
                f"{dim1:+.6f}"
            )

            print(
                f"Centered signal: "
                f"{signal:+.6f}"
            )

            if (
                abs(signal)
                < args.deadband
            ):
                delta = 0.0

                print(
                    "Signal inside deadband."
                )

            else:
                delta = (
                    mapping_sign
                    * gain
                    * signal
                )

                delta = float(
                    np.clip(
                        delta,
                        -args.max_step_deg,
                        args.max_step_deg,
                    )
                )

            print(
                f"Proposed delta J1: "
                f"{delta:+.3f} deg"
            )

            target_j1 = float(
                np.clip(
                    state[0]
                    + delta,
                    min_allowed,
                    max_allowed,
                )
            )

            print(
                f"Proposed target J1: "
                f"{target_j1:.3f} deg"
            )

            if not args.execute:
                print(
                    "DRY RUN: no motion executed."
                )

                break

            if abs(
                target_j1
                - state[0]
            ) < 0.05:
                print(
                    "STOP: command too small."
                )
                break

            if robot.mc is None:
                raise RuntimeError(
                    "Robot controller unavailable."
                )

            robot.mc.send_angle(
                1,
                target_j1,
                args.speed,
            )

            time.sleep(
                args.wait
            )

            new_angles = (
                robot.mc.get_angles()
            )

            print(
                f"Actual J1 after command: "
                f"{new_angles[0]:.3f} deg"
            )

            if (
                previous_pixel_error
                is not None
            ):
                if (
                    abs(pixel_error)
                    >
                    abs(
                        previous_pixel_error
                    )
                    + 15.0
                ):
                    worsening_count += 1
                else:
                    worsening_count = 0

                if worsening_count >= 2:
                    print()
                    print(
                        "SAFETY STOP:"
                    )

                    print(
                        "Visual error worsened "
                        "for two consecutive steps."
                    )

                    break

            previous_pixel_error = (
                pixel_error
            )

        print()
        print("=" * 70)
        print(
            "CLOSED LOOP FINISHED"
        )
        print("=" * 70)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()