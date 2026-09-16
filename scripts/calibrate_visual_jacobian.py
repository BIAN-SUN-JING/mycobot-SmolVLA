import argparse
import json
import sys
import time
from datetime import datetime
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
    / "visual_jacobian"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "visual_jacobian.json"
)


JOINT_IDS = [1, 2, 3]

JOINT_NAMES = [
    "J1",
    "J2",
    "J3",
]


# myCobot 280 approximate joint limits
JOINT_LIMITS = {
    1: (-168.0, 168.0),
    2: (-140.0, 140.0),
    3: (-150.0, 150.0),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Requested +/- joint perturbation in degrees.",
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--wait",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--min-red-area",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--min-observed-span",
        type=float,
        default=0.8,
        help=(
            "Minimum actual +q to -q joint span "
            "required for calibration."
        ),
    )

    return parser.parse_args()


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


def detect_red(
    frame,
    min_red_area,
):
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
                "mask": mask,
                "mode": color_mode,
            }
        )

    if not candidates:
        raise RuntimeError(
            "Red target was not detected."
        )

    return max(
        candidates,
        key=lambda x: x["area"],
    )


def save_detection(
    frame,
    detection,
    filename,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        (
            f"u={cx}, v={cy}, "
            f"area={detection['area']:.0f}"
        ),
        (
            max(5, cx - 120),
            max(20, cy - 15),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    cv2.imwrite(
        str(path),
        image,
    )

    return path


def read_angles(robot):
    if robot.mc is None:
        raise RuntimeError(
            "Robot controller unavailable."
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


def read_visual_feature(
    robot,
    min_red_area,
    label,
):
    frame = (
        robot.cameras[
            "camera1"
        ]
        .async_read(
            timeout_ms=2000
        )
        .copy()
    )

    detection = detect_red(
        frame,
        min_red_area,
    )

    feature = np.asarray(
        [
            detection["cx"],
            detection["cy"],
            np.log(
                max(
                    detection["area"],
                    1.0,
                )
            ),
        ],
        dtype=np.float64,
    )

    image_path = save_detection(
        frame,
        detection,
        f"{label}.jpg",
    )

    print(
        f"[{label}] "
        f"u={feature[0]:.2f}, "
        f"v={feature[1]:.2f}, "
        f"area={detection['area']:.1f}, "
        f"log(area)={feature[2]:.4f}"
    )

    return {
        "feature": feature,
        "area": float(
            detection["area"]
        ),
        "image": str(
            image_path
        ),
    }


def move_joint(
    robot,
    joint_id,
    target,
    speed,
    wait_seconds,
):
    if robot.mc is None:
        raise RuntimeError(
            "Robot controller unavailable."
        )

    low, high = (
        JOINT_LIMITS[
            joint_id
        ]
    )

    target = float(
        np.clip(
            target,
            low,
            high,
        )
    )

    before = read_angles(
        robot
    )

    print(
        f"[Motion] J{joint_id}: "
        f"{before[joint_id - 1]:.2f} "
        f"-> {target:.2f}"
    )

    robot.mc.send_angle(
        joint_id,
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
        f"[Motion] actual J{joint_id}: "
        f"{after[joint_id - 1]:.2f}"
    )

    return after


def return_joint(
    robot,
    joint_id,
    baseline_angle,
    speed,
    wait_seconds,
):
    print(
        f"Returning J{joint_id} "
        f"to baseline..."
    )

    return move_joint(
        robot,
        joint_id,
        baseline_angle,
        speed,
        wait_seconds,
    )


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = (
        MyCobot280RobotConfig(
            id="visual_jacobian_calibration",
            port="auto",
            baudrate=1_000_000,
            enable_motion=False,
        )
    )

    robot = MyCobot280Robot(
        config
    )

    baseline_angles = None

    report = {
        "timestamp": (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        ),

        "joint_ids":
            JOINT_IDS,

        "joint_names":
            JOINT_NAMES,

        "visual_feature": [
            "u_pixel",
            "v_pixel",
            "log_area",
        ],

        "requested_delta_deg":
            float(
                args.delta
            ),

        "columns": {},
    }

    try:
        print()
        print("=" * 72)
        print(
            "myCobot VISUAL JACOBIAN CALIBRATION"
        )
        print("=" * 72)

        print()
        print(
            "This test will perturb only J1, J2, J3."
        )

        print(
            f"Requested perturbation: "
            f"+/-{args.delta:.2f} deg"
        )

        print()
        print(
            "KEEP THE RED OBJECT COMPLETELY STATIONARY."
        )

        print(
            "Make sure the robot has free space "
            "for small motions."
        )

        print()
        input(
            "Press ENTER when the workspace is safe..."
        )

        robot.connect(
            calibrate=False
        )

        if robot.mc is None:
            raise RuntimeError(
                "Controller unavailable."
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

        baseline_angles = (
            read_angles(
                robot
            )
        )

        report[
            "baseline_angles_deg"
        ] = [
            float(x)
            for x in baseline_angles
        ]

        print()
        print(
            "Baseline:"
        )

        print(
            [
                round(
                    float(x),
                    3,
                )
                for x in baseline_angles
            ]
        )

        baseline_visual = (
            read_visual_feature(
                robot,
                args.min_red_area,
                "baseline",
            )
        )

        report[
            "baseline_visual_feature"
        ] = baseline_visual[
            "feature"
        ].tolist()

        jacobian_columns = []

        # ====================================================
        # Calibrate J1/J2/J3
        # ====================================================

        for joint_id in JOINT_IDS:
            print()
            print("=" * 72)

            print(
                f"CALIBRATING J{joint_id}"
            )

            print("=" * 72)

            baseline_joint = float(
                baseline_angles[
                    joint_id - 1
                ]
            )

            # -----------------------------------------------
            # Positive perturbation
            # -----------------------------------------------

            positive_target = (
                baseline_joint
                + args.delta
            )

            positive_angles = (
                move_joint(
                    robot,
                    joint_id,
                    positive_target,
                    args.speed,
                    args.wait,
                )
            )

            positive_visual = (
                read_visual_feature(
                    robot,
                    args.min_red_area,
                    f"J{joint_id}_positive",
                )
            )

            # -----------------------------------------------
            # Return
            # -----------------------------------------------

            return_joint(
                robot,
                joint_id,
                baseline_joint,
                args.speed,
                args.wait,
            )

            # -----------------------------------------------
            # Negative perturbation
            # -----------------------------------------------

            negative_target = (
                baseline_joint
                - args.delta
            )

            negative_angles = (
                move_joint(
                    robot,
                    joint_id,
                    negative_target,
                    args.speed,
                    args.wait,
                )
            )

            negative_visual = (
                read_visual_feature(
                    robot,
                    args.min_red_area,
                    f"J{joint_id}_negative",
                )
            )

            # -----------------------------------------------
            # Return
            # -----------------------------------------------

            return_joint(
                robot,
                joint_id,
                baseline_joint,
                args.speed,
                args.wait,
            )

            # -----------------------------------------------
            # Central finite difference
            # -----------------------------------------------

            q_plus = float(
                positive_angles[
                    joint_id - 1
                ]
            )

            q_minus = float(
                negative_angles[
                    joint_id - 1
                ]
            )

            observed_span = (
                q_plus
                - q_minus
            )

            print()
            print(
                f"Observed J{joint_id} span: "
                f"{observed_span:+.3f} deg"
            )

            if (
                abs(observed_span)
                < args.min_observed_span
            ):
                raise RuntimeError(
                    f"J{joint_id} moved too little "
                    f"for reliable calibration. "
                    f"Observed span="
                    f"{observed_span:.3f} deg. "
                    f"Try --delta 3.0."
                )

            s_plus = (
                positive_visual[
                    "feature"
                ]
            )

            s_minus = (
                negative_visual[
                    "feature"
                ]
            )

            column = (
                s_plus
                - s_minus
            ) / observed_span

            jacobian_columns.append(
                column
            )

            print()
            print(
                f"J{joint_id} column:"
            )

            print(
                f"  du/dq = "
                f"{column[0]:+.4f} "
                f"px/deg"
            )

            print(
                f"  dv/dq = "
                f"{column[1]:+.4f} "
                f"px/deg"
            )

            print(
                f"  dlogA/dq = "
                f"{column[2]:+.6f} "
                f"/deg"
            )

            report[
                "columns"
            ][
                f"J{joint_id}"
            ] = {
                "q_plus_deg":
                    q_plus,

                "q_minus_deg":
                    q_minus,

                "observed_span_deg":
                    observed_span,

                "s_plus":
                    s_plus.tolist(),

                "s_minus":
                    s_minus.tolist(),

                "column":
                    column.tolist(),

                "positive_image":
                    positive_visual[
                        "image"
                    ],

                "negative_image":
                    negative_visual[
                        "image"
                    ],
            }

        # ====================================================
        # Build Jacobian
        # ====================================================

        J = np.column_stack(
            jacobian_columns
        )

        print()
        print("=" * 72)

        print(
            "VISUAL JACOBIAN"
        )

        print("=" * 72)

        print()
        print(
            "Rows:"
        )

        print(
            "  [u, v, log(area)]"
        )

        print(
            "Columns:"
        )

        print(
            "  [J1, J2, J3]"
        )

        print()

        print(
            np.array2string(
                J,
                precision=6,
                suppress_small=False,
            )
        )

        # ====================================================
        # Condition diagnostics
        # ====================================================

        singular_values = (
            np.linalg.svd(
                J,
                compute_uv=False,
            )
        )

        condition_number = float(
            np.linalg.cond(
                J
            )
        )

        rank = int(
            np.linalg.matrix_rank(
                J
            )
        )

        J_pinv = np.linalg.pinv(
            J
        )

        print()
        print(
            "rank:",
            rank,
        )

        print(
            "singular values:",
            [
                round(
                    float(x),
                    6,
                )
                for x in singular_values
            ],
        )

        print(
            "condition number:",
            condition_number,
        )

        print()
        print(
            "Pseudo-inverse:"
        )

        print(
            np.array2string(
                J_pinv,
                precision=6,
                suppress_small=False,
            )
        )

        report[
            "jacobian"
        ] = J.tolist()

        report[
            "pseudo_inverse"
        ] = (
            J_pinv.tolist()
        )

        report[
            "rank"
        ] = rank

        report[
            "singular_values"
        ] = [
            float(x)
            for x in singular_values
        ]

        report[
            "condition_number"
        ] = (
            condition_number
        )

        # ====================================================
        # Verdict
        # ====================================================

        if rank < 3:
            verdict = (
                "RANK_DEFICIENT"
            )

        elif condition_number > 1000:
            verdict = (
                "VERY_ILL_CONDITIONED"
            )

        elif condition_number > 100:
            verdict = (
                "ILL_CONDITIONED"
            )

        else:
            verdict = (
                "USABLE_LOCAL_JACOBIAN"
            )

        report[
            "verdict"
        ] = verdict

        print()
        print("=" * 72)

        print(
            "VERDICT"
        )

        print("=" * 72)

        print(
            verdict
        )

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
        print(
            "VISUAL_JACOBIAN_CALIBRATION_OK"
        )

    finally:
        # ====================================================
        # Safety return of J1-J3
        # ====================================================

        if (
            robot.mc is not None
            and baseline_angles
            is not None
        ):
            try:
                current = read_angles(
                    robot
                )

                for joint_id in JOINT_IDS:
                    baseline_joint = float(
                        baseline_angles[
                            joint_id - 1
                        ]
                    )

                    if (
                        abs(
                            current[
                                joint_id - 1
                            ]
                            - baseline_joint
                        )
                        > 0.5
                    ):
                        print()
                        print(
                            f"[Safety] Returning "
                            f"J{joint_id} to baseline..."
                        )

                        robot.mc.send_angle(
                            joint_id,
                            baseline_joint,
                            args.speed,
                        )

                        time.sleep(
                            args.wait
                        )

            except Exception as exc:
                print()
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