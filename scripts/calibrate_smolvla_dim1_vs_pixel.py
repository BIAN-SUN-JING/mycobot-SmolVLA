import argparse
import json
import sys
import time
from datetime import datetime
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

LEGACY_STATS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "smolvla_stats"
    / "legacy_smolvla_normalization_stats.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dim1_pixel_calibration"
)

JOINT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
]

SO100_STATE_MEAN_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.mean"
)

SO100_STATE_STD_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.std"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples-per-position",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min-red-area",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--center-x",
        type=float,
        default=320.0,
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_state_stats():
    data = load_json(
        LEGACY_STATS_FILE
    )

    stats = data["stats"]

    mean = torch.tensor(
        stats[
            SO100_STATE_MEAN_KEY
        ]["values"],
        dtype=torch.float32,
    )

    std = torch.tensor(
        stats[
            SO100_STATE_STD_KEY
        ]["values"],
        dtype=torch.float32,
    )

    return mean, std


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

    mask = cv2.bitwise_or(
        mask1,
        mask2,
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def detect_red(frame, min_red_area):
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

        if area < min_red_area:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = float(
            moments["m10"]
            / moments["m00"]
        )

        cy = float(
            moments["m01"]
            / moments["m00"]
        )

        candidates.append(
            {
                "cx": cx,
                "cy": cy,
                "area": area,
                "mode": mode,
                "mask": mask,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["area"],
    )


def save_detection_image(
    frame,
    detection,
    path,
):
    if detection["mode"] == "RGB":
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )
    else:
        image = frame.copy()

    cx = int(
        round(
            detection["cx"]
        )
    )

    cy = int(
        round(
            detection["cy"]
        )
    )

    cv2.circle(
        image,
        (cx, cy),
        8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        f"x={cx}",
        (
            max(5, cx - 40),
            max(20, cy - 15),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(path),
        image,
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
        batch["observation.state"]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    mean = state_mean.to(
        device=device,
        dtype=torch.float32,
    )

    std = state_std.to(
        device=device,
        dtype=torch.float32,
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

    return float(
        action[0]
    )


def fit_line(xs, ys):
    xs = np.asarray(
        xs,
        dtype=np.float64,
    )

    ys = np.asarray(
        ys,
        dtype=np.float64,
    )

    coeffs = np.polyfit(
        xs,
        ys,
        deg=1,
    )

    slope = float(
        coeffs[0]
    )

    intercept = float(
        coeffs[1]
    )

    y_pred = (
        slope * xs
        + intercept
    )

    ss_res = float(
        np.sum(
            (
                ys - y_pred
            ) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                ys - ys.mean()
            ) ** 2
        )
    )

    if ss_tot <= 1e-12:
        r2 = 0.0
    else:
        r2 = (
            1.0
            - ss_res / ss_tot
        )

    return (
        slope,
        intercept,
        r2,
    )


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    state_mean, state_std = (
        load_state_stats()
    )

    print()
    print("=" * 72)
    print(
        "SmolVLA DIM1 vs PIXEL-X CALIBRATION"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    print(
        "Task:",
        TASK,
    )

    print(
        "Samples per position:",
        args.samples_per_position,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The robot will NOT move."
    )

    print(
        "Only move the red object by hand."
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
            id="dim1_pixel_calibration",
            port="auto",
            baudrate=1_000_000,
            enable_motion=False,
        )
    )

    robot = MyCobot280Robot(
        config
    )

    records = []

    try:
        robot.connect(
            calibrate=False
        )

        dataset_features = (
            build_features(
                robot
            )
        )

        print()
        print(
            "ROBOT_CONNECTED"
        )

        print()
        print(
            "Suggested positions:"
        )

        print(
            "  P1 ~ x=100"
        )

        print(
            "  P2 ~ x=210"
        )

        print(
            "  P3 ~ x=320"
        )

        print(
            "  P4 ~ x=430"
        )

        print(
            "  P5 ~ x=540"
        )

        print()

        for position_index in range(
            1,
            6,
        ):
            print()
            print("=" * 72)

            print(
                f"POSITION P{position_index}"
            )

            print("=" * 72)

            input(
                "Place the RED object at this position, "
                "wait 2 seconds, then press ENTER..."
            )

            time.sleep(2.0)

            position_samples = []

            for sample_index in range(
                1,
                args.samples_per_position + 1,
            ):
                observation = (
                    robot.get_observation()
                )

                frame = observation[
                    "camera1"
                ]

                detection = detect_red(
                    frame,
                    args.min_red_area,
                )

                if detection is None:
                    raise RuntimeError(
                        "Red object not detected."
                    )

                dim1 = infer_dim1(
                    policy=policy,
                    preprocessor=preprocessor,
                    observation=observation,
                    dataset_features=dataset_features,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    seed=args.seed,
                )

                timestamp = (
                    datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )

                image_path = (
                    OUTPUT_DIR
                    / (
                        f"P{position_index}_"
                        f"S{sample_index}_"
                        f"{timestamp}.jpg"
                    )
                )

                save_detection_image(
                    frame,
                    detection,
                    image_path,
                )

                sample = {
                    "position":
                        position_index,

                    "sample":
                        sample_index,

                    "pixel_x":
                        float(
                            detection["cx"]
                        ),

                    "pixel_y":
                        float(
                            detection["cy"]
                        ),

                    "red_area":
                        float(
                            detection["area"]
                        ),

                    "dim1":
                        float(dim1),

                    "image":
                        str(image_path),
                }

                records.append(
                    sample
                )

                position_samples.append(
                    sample
                )

                print(
                    f"P{position_index} "
                    f"S{sample_index}: "
                    f"x="
                    f"{detection['cx']:.2f}px, "
                    f"dim1="
                    f"{dim1:+.6f}"
                )

            mean_x = np.mean(
                [
                    item["pixel_x"]
                    for item
                    in position_samples
                ]
            )

            mean_dim1 = np.mean(
                [
                    item["dim1"]
                    for item
                    in position_samples
                ]
            )

            std_dim1 = np.std(
                [
                    item["dim1"]
                    for item
                    in position_samples
                ]
            )

            print()
            print(
                f"P{position_index} mean:"
            )

            print(
                f"  pixel_x = "
                f"{mean_x:.2f}"
            )

            print(
                f"  dim1    = "
                f"{mean_dim1:+.6f}"
            )

            print(
                f"  dim1 std= "
                f"{std_dim1:.6f}"
            )

        # ----------------------------------------------------
        # Fit using all samples
        # ----------------------------------------------------

        xs = [
            item["pixel_x"]
            for item in records
        ]

        ys = [
            item["dim1"]
            for item in records
        ]

        slope, intercept, r2 = (
            fit_line(
                xs,
                ys,
            )
        )

        center_dim1 = (
            slope * args.center_x
            + intercept
        )

        print()
        print("=" * 72)
        print(
            "CALIBRATION RESULT"
        )
        print("=" * 72)

        print()
        print(
            "Linear model:"
        )

        print(
            f"dim1 = "
            f"{slope:+.8f} * pixel_x "
            f"{intercept:+.8f}"
        )

        print()
        print(
            f"R^2 = {r2:.4f}"
        )

        print(
            f"Slope = "
            f"{slope:+.8f} "
            f"dim1/pixel"
        )

        print(
            f"Neutral dim1 at "
            f"x={args.center_x:.1f}: "
            f"{center_dim1:+.6f}"
        )

        # ----------------------------------------------------
        # Group means
        # ----------------------------------------------------

        group_summaries = []

        for position_index in range(
            1,
            6,
        ):
            subset = [
                item
                for item in records
                if item["position"]
                == position_index
            ]

            group_summaries.append(
                {
                    "position":
                        position_index,

                    "mean_pixel_x":
                        float(
                            np.mean(
                                [
                                    item["pixel_x"]
                                    for item
                                    in subset
                                ]
                            )
                        ),

                    "mean_dim1":
                        float(
                            np.mean(
                                [
                                    item["dim1"]
                                    for item
                                    in subset
                                ]
                            )
                        ),

                    "std_dim1":
                        float(
                            np.std(
                                [
                                    item["dim1"]
                                    for item
                                    in subset
                                ]
                            )
                        ),
                }
            )

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        if r2 >= 0.80:
            verdict = (
                "STRONG_LINEAR_RELATION"
            )

        elif r2 >= 0.50:
            verdict = (
                "MODERATE_LINEAR_RELATION"
            )

        else:
            verdict = (
                "WEAK_OR_NONLINEAR_RELATION"
            )

        print()
        print(
            "Verdict:",
            verdict,
        )

        print()
        print(
            "No robot action was executed."
        )

        report = {
            "model":
                MODEL_ID,

            "task":
                TASK,

            "seed":
                args.seed,

            "center_x":
                args.center_x,

            "samples":
                records,

            "groups":
                group_summaries,

            "fit": {
                "slope":
                    slope,

                "intercept":
                    intercept,

                "r2":
                    r2,

                "neutral_dim1_at_center":
                    center_dim1,
            },

            "verdict":
                verdict,
        }

        report_path = (
            OUTPUT_DIR
            / "dim1_pixel_calibration.json"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            "Report saved:"
        )

        print(
            report_path
        )

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()