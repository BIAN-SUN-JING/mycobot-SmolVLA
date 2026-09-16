import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


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
    / "j1_visual_calibration"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "j1_visual_sign.json"
)


# Derived from your clean 3x3 evaluation.
RIGHT_TOWARD_DIM1_MEAN = 0.38705406
LEFT_TOWARD_DIM1_MEAN = 0.13109098

DIM1_NEUTRAL = (
    RIGHT_TOWARD_DIM1_MEAN
    + LEFT_TOWARD_DIM1_MEAN
) / 2.0

DIM1_RIGHT_AMPLITUDE = (
    RIGHT_TOWARD_DIM1_MEAN
    - DIM1_NEUTRAL
)


J1_MIN = -168.0
J1_MAX = 168.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help=(
            "Requested J1 calibration displacement "
            "in degrees."
        ),
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--wait",
        type=float,
        default=2.5,
    )

    return parser.parse_args()


def read_angles(robot):
    if robot.mc is None:
        raise RuntimeError(
            "myCobot controller unavailable."
        )

    angles = robot.mc.get_angles()

    if (
        angles is None
        or len(angles) != 6
    ):
        raise RuntimeError(
            f"Invalid joint angles: {angles}"
        )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


def read_frame(robot):
    camera = robot.cameras[
        "camera1"
    ]

    return camera.async_read(
        timeout_ms=2000
    ).copy()


def make_red_mask(hsv):
    lower_red_1 = np.array(
        [0, 80, 60],
        dtype=np.uint8,
    )

    upper_red_1 = np.array(
        [12, 255, 255],
        dtype=np.uint8,
    )

    lower_red_2 = np.array(
        [168, 80, 60],
        dtype=np.uint8,
    )

    upper_red_2 = np.array(
        [180, 255, 255],
        dtype=np.uint8,
    )

    mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1,
    )

    mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2,
    )

    mask = cv2.bitwise_or(
        mask_1,
        mask_2,
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
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


def find_largest_red_region(
    frame,
):
    """
    LeRobot/OpenCV color convention may differ
    depending on camera backend.

    Try both RGB and BGR interpretations and
    keep the detection with the largest red area.
    """

    candidates = []

    for color_mode in (
        "RGB",
        "BGR",
    ):
        if color_mode == "RGB":
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

        contours, _ = (
            cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
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
            {
                "cx": float(cx),
                "cy": float(cy),
                "area": area,
                "mask": mask,
                "contour": contour,
                "color_mode":
                    color_mode,
            }
        )

    if not candidates:
        raise RuntimeError(
            "Red object was not detected. "
            "Make sure the red target is clearly visible "
            "in the wrist camera."
        )

    result = max(
        candidates,
        key=lambda x:
            x["area"],
    )

    return result


def frame_for_saving(
    frame,
    color_mode,
):
    if color_mode == "RGB":
        return cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )

    return frame.copy()


def save_detection(
    frame,
    detection,
    name,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = frame_for_saving(
        frame,
        detection[
            "color_mode"
        ],
    )

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
        f"red=({cx},{cy})",
        (
            max(
                5,
                cx - 80,
            ),
            max(
                20,
                cy - 15,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    image_path = (
        OUTPUT_DIR
        / f"{name}.jpg"
    )

    mask_path = (
        OUTPUT_DIR
        / f"{name}_mask.png"
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    cv2.imwrite(
        str(mask_path),
        detection[
            "mask"
        ],
    )

    return (
        image_path,
        mask_path,
    )


def move_j1(
    robot,
    target,
    speed,
    wait_seconds,
):
    if robot.mc is None:
        raise RuntimeError(
            "Robot controller unavailable."
        )

    target = float(
        np.clip(
            target,
            J1_MIN,
            J1_MAX,
        )
    )

    before = read_angles(
        robot
    )

    print()
    print(
        f"[Motion] J1 "
        f"{before[0]:.2f} "
        f"-> "
        f"{target:.2f}"
    )

    robot.mc.send_angle(
        1,
        target,
        speed,
    )

    time.sleep(
        wait_seconds
    )

    after = read_angles(
        robot
    )

    print(
        f"[Motion] actual J1 = "
        f"{after[0]:.2f}"
    )

    return after


def capture_sample(
    robot,
    label,
):
    angles = read_angles(
        robot
    )

    frame = read_frame(
        robot
    )

    detection = (
        find_largest_red_region(
            frame
        )
    )

    image_path, mask_path = (
        save_detection(
            frame,
            detection,
            label,
        )
    )

    print()
    print(
        f"[{label}]"
    )

    print(
        f"J1 = "
        f"{angles[0]:.3f} deg"
    )

    print(
        f"red centroid = "
        f"({detection['cx']:.2f}, "
        f"{detection['cy']:.2f})"
    )

    print(
        f"red area = "
        f"{detection['area']:.1f}"
    )

    print(
        f"color mode = "
        f"{detection['color_mode']}"
    )

    return {
        "angles":
            angles,

        "frame":
            frame,

        "cx":
            detection["cx"],

        "cy":
            detection["cy"],

        "area":
            detection["area"],

        "image_path":
            str(image_path),

        "mask_path":
            str(mask_path),
    }


def main():
    args = parse_args()

    delta = float(
        args.delta
    )

    speed = int(
        args.speed
    )

    wait_seconds = float(
        args.wait
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = (
        MyCobot280RobotConfig(
            id="j1_visual_calibration",
            port="auto",
            baudrate=1_000_000,

            # Motion is explicitly enabled only
            # for this controlled calibration.
            enable_motion=True,
        )
    )

    robot = MyCobot280Robot(
        config
    )

    baseline_j1 = None

    try:
        print()
        print("=" * 70)
        print(
            "J1 WRIST-CAMERA VISUAL SIGN CALIBRATION"
        )
        print("=" * 70)

        print()
        print(
            "This test will move ONLY J1."
        )

        print(
            f"Requested displacement: "
            f"+/- {delta:.2f} deg"
        )

        print(
            f"Speed: {speed}"
        )

        print()
        print(
            "Keep the red object completely stationary."
        )

        print(
            "Make sure the arm has free space "
            "to rotate a few degrees."
        )

        print()
        input(
            "Press ENTER when the workspace is safe..."
        )

        robot.connect(
            calibrate=False
        )

        print()
        print(
            "ROBOT_CONNECTED"
        )

        # ====================================================
        # Baseline
        # ====================================================

        baseline = capture_sample(
            robot,
            "baseline",
        )

        baseline_j1 = float(
            baseline[
                "angles"
            ][0]
        )

        print()
        print(
            f"Baseline J1 = "
            f"{baseline_j1:.3f}"
        )

        # ====================================================
        # Positive J1
        # ====================================================

        positive_target = (
            baseline_j1
            + delta
        )

        move_j1(
            robot,
            positive_target,
            speed,
            wait_seconds,
        )

        positive = capture_sample(
            robot,
            "j1_positive",
        )

        # ====================================================
        # Return to baseline
        # ====================================================

        print()
        print(
            "Returning to baseline..."
        )

        move_j1(
            robot,
            baseline_j1,
            speed,
            wait_seconds,
        )

        # ====================================================
        # Negative J1
        # ====================================================

        negative_target = (
            baseline_j1
            - delta
        )

        move_j1(
            robot,
            negative_target,
            speed,
            wait_seconds,
        )

        negative = capture_sample(
            robot,
            "j1_negative",
        )

        # ====================================================
        # Return
        # ====================================================

        print()
        print(
            "Returning to baseline..."
        )

        final_angles = move_j1(
            robot,
            baseline_j1,
            speed,
            wait_seconds,
        )

        # ====================================================
        # Geometry analysis
        # ====================================================

        j_plus = float(
            positive[
                "angles"
            ][0]
        )

        j_minus = float(
            negative[
                "angles"
            ][0]
        )

        x_plus = float(
            positive["cx"]
        )

        x_minus = float(
            negative["cx"]
        )

        x_base = float(
            baseline["cx"]
        )

        joint_span = (
            j_plus
            - j_minus
        )

        pixel_span = (
            x_plus
            - x_minus
        )

        if abs(
            joint_span
        ) < 0.5:
            raise RuntimeError(
                "J1 did not move enough for reliable "
                "visual calibration. "
                "Re-run with --delta 3.0."
            )

        dx_per_deg = (
            pixel_span
            / joint_span
        )

        if abs(
            dx_per_deg
        ) < 0.5:
            raise RuntimeError(
                "The red object's image position changed "
                "too little to determine J1 sign reliably."
            )

        # SmolVLA dim1:
        # right target -> centered signal > 0
        # left target  -> centered signal < 0
        #
        # To reduce positive image-x error (object right),
        # desired joint direction must be opposite to
        # sign(dx/dJ1).
        mapping_sign = int(
            -np.sign(
                dx_per_deg
            )
        )

        print()
        print("=" * 70)
        print(
            "CALIBRATION RESULT"
        )
        print("=" * 70)

        print()
        print(
            f"Baseline centroid x: "
            f"{x_base:.2f}px"
        )

        print(
            f"J1 positive centroid x: "
            f"{x_plus:.2f}px"
        )

        print(
            f"J1 negative centroid x: "
            f"{x_minus:.2f}px"
        )

        print()
        print(
            f"Observed J1+: "
            f"{j_plus:.3f} deg"
        )

        print(
            f"Observed J1-: "
            f"{j_minus:.3f} deg"
        )

        print()
        print(
            f"dx / dJ1 = "
            f"{dx_per_deg:+.3f} "
            f"pixels/degree"
        )

        if dx_per_deg < 0:
            print(
                "Positive J1 moves the red object "
                "LEFT in the wrist-camera image."
            )
        else:
            print(
                "Positive J1 moves the red object "
                "RIGHT in the wrist-camera image."
            )

        print()
        print(
            "Recommended SmolVLA dim1 -> "
            f"myCobot J1 sign: "
            f"{mapping_sign:+d}"
        )

        print()
        print(
            f"SmolVLA dim1 neutral = "
            f"{DIM1_NEUTRAL:.6f}"
        )

        print(
            f"Typical right/left amplitude = "
            f"{DIM1_RIGHT_AMPLITUDE:.6f}"
        )

        report = {
            "baseline_j1_deg":
                baseline_j1,

            "final_j1_deg":
                float(
                    final_angles[0]
                ),

            "requested_delta_deg":
                delta,

            "observed_j1_positive_deg":
                j_plus,

            "observed_j1_negative_deg":
                j_minus,

            "baseline_centroid_x":
                x_base,

            "positive_centroid_x":
                x_plus,

            "negative_centroid_x":
                x_minus,

            "dx_per_j1_degree":
                float(
                    dx_per_deg
                ),

            "smolvla_dim1_neutral":
                DIM1_NEUTRAL,

            "smolvla_dim1_typical_amplitude":
                DIM1_RIGHT_AMPLITUDE,

            "recommended_mapping_sign":
                mapping_sign,

            "future_formula": (
                "delta_J1 = "
                "mapping_sign * gain * "
                "(smolvla_dim1 - neutral)"
            ),

            "robot_action_executed":
                False,
        }

        with REPORT_PATH.open(
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
            REPORT_PATH
        )

        print()
        print("=" * 70)
        print(
            "J1_VISUAL_SIGN_CALIBRATION_OK"
        )
        print("=" * 70)

    finally:
        # Return to baseline whenever possible.
        if (
            robot.mc is not None
            and baseline_j1
            is not None
        ):
            try:
                current = (
                    robot.mc
                    .get_angles()
                )

                if (
                    current is not None
                    and len(current) == 6
                    and abs(
                        current[0]
                        - baseline_j1
                    ) > 0.5
                ):
                    print()
                    print(
                        "[Safety] Returning J1 "
                        "to baseline..."
                    )

                    robot.mc.send_angle(
                        1,
                        baseline_j1,
                        speed,
                    )

                    time.sleep(
                        wait_seconds
                    )

            except Exception as exc:
                print(
                    "[Safety warning]",
                    repr(exc),
                )

        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()