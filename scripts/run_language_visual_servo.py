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


# ============================================================
# Files
# ============================================================

JACOBIAN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "visual_jacobian"
    / "visual_jacobian.json"
)

STOP_REFERENCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "red_stop_reference"
    / "red_stop_reference.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "language_visual_servo"
)


# ============================================================
# Robot limits
# ============================================================

JOINT_LIMITS = {
    1: (-168.0, 168.0),
    2: (-140.0, 140.0),
    3: (-150.0, 150.0),
}


# ============================================================
# Utilities
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        type=str,
        default="Move to the red object.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move J1/J2/J3.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--wait",
        type=float,
        default=1.3,
    )

    parser.add_argument(
        "--center-tolerance-px",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--approach-center-limit-px",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--center-gain",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--approach-step-deg",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--area-fraction",
        type=float,
        default=0.65,
        help=(
            "Stop at this fraction of the manually "
            "measured safe reference area."
        ),
    )

    parser.add_argument(
        "--max-total-joint-travel",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--start-pose-tolerance",
        type=float,
        default=6.0,
        help=(
            "Maximum J1-J3 difference from Jacobian "
            "calibration pose when --execute is used."
        ),
    )

    return parser.parse_args()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Language layer
# ============================================================

def parse_instruction(instruction):
    text = instruction.lower()

    if "red" not in text:
        raise ValueError(
            "Current version only supports a red target."
        )

    if not any(
        word in text
        for word in (
            "move",
            "go",
            "approach",
            "toward",
        )
    ):
        raise ValueError(
            "Instruction does not request motion "
            "toward the target."
        )

    return {
        "target_color": "red",
        "operation": "approach",
    }


# ============================================================
# Vision
# ============================================================

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


def detect_red(frame):
    candidates = []

    for mode in ("RGB", "BGR"):
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

        mask = make_red_mask(hsv)

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
            cv2.contourArea(contour)
        )

        if area < 100:
            continue

        moments = cv2.moments(contour)

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
                "u": cx,
                "v": cy,
                "area": area,
                "mode": mode,
                "contour": contour,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["area"],
    )


def save_debug_frame(
    frame,
    detection,
    target_u,
    target_v,
    step_index,
    phase,
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

    u = int(round(detection["u"]))
    v = int(round(detection["v"]))

    tu = int(round(target_u))
    tv = int(round(target_v))

    cv2.circle(
        image,
        (u, v),
        8,
        (0, 255, 0),
        2,
    )

    cv2.circle(
        image,
        (tu, tv),
        10,
        (255, 0, 255),
        2,
    )

    cv2.line(
        image,
        (u, v),
        (tu, tv),
        (0, 255, 255),
        2,
    )

    text = (
        f"{phase} "
        f"area={detection['area']:.0f}"
    )

    cv2.putText(
        image,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    path = (
        OUTPUT_DIR
        / f"step_{step_index:03d}_{phase}.jpg"
    )

    cv2.imwrite(
        str(path),
        image,
    )

    return path


# ============================================================
# Robot
# ============================================================

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
            f"Invalid joint state: {angles}"
        )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


def read_camera(robot):
    return (
        robot.cameras["camera1"]
        .async_read(
            timeout_ms=2000
        )
        .copy()
    )


def execute_delta(
    robot,
    dq,
    speed,
    wait_seconds,
):
    """
    Execute J1/J2/J3 sequentially.

    Each joint is verified immediately after its command.
    If one joint fails to move, later joints will NOT be
    commanded.

    J4-J6 are never commanded.
    """

    if robot.mc is None:
        raise RuntimeError(
            "Robot controller unavailable."
        )

    initial = read_angles(
        robot
    )

    targets = initial[:3].copy()

    for i in range(3):
        joint_id = i + 1

        low, high = (
            JOINT_LIMITS[
                joint_id
            ]
        )

        targets[i] = float(
            np.clip(
                initial[i] + dq[i],
                low,
                high,
            )
        )

    print(
        "Command:"
    )

    for i in range(3):
        print(
            f"  J{i + 1}: "
            f"{initial[i]:8.3f} "
            f"-> {targets[i]:8.3f} "
            f"(Δ={targets[i]-initial[i]:+7.3f})"
        )

    # ========================================================
    # Execute and verify each joint independently
    # ========================================================

    for i in range(3):
        joint_id = i + 1

        before = read_angles(
            robot
        )

        requested_delta = (
            targets[i]
            - before[i]
        )

        # Skip very small commands.
        if abs(requested_delta) < 0.15:
            continue

        print()
        print(
            f"[Execute] J{joint_id} "
            f"{before[i]:.3f} "
            f"-> {targets[i]:.3f}"
        )

        robot.mc.send_angle(
            joint_id,
            float(targets[i]),
            speed,
        )

        # Give this servo time to react before checking.
        time.sleep(0.8)

        after = read_angles(
            robot
        )

        observed_delta = (
            after[i]
            - before[i]
        )

        print(
            f"[Verify] J{joint_id}: "
            f"requested={requested_delta:+.3f}°, "
            f"observed={observed_delta:+.3f}°"
        )

        # Motion should at least be detectable.
        if abs(observed_delta) < 0.20:
            raise RuntimeError(
                "MOTION_NOT_EXECUTED: "
                f"J{joint_id} requested "
                f"{requested_delta:+.3f} deg, "
                f"but observed only "
                f"{observed_delta:+.3f} deg."
            )

        # Also check direction.
        if (
            np.sign(observed_delta)
            != np.sign(requested_delta)
        ):
            raise RuntimeError(
                "MOTION_DIRECTION_ERROR: "
                f"J{joint_id} moved in the "
                f"wrong direction."
            )

    # Allow the complete robot configuration to settle.
    time.sleep(
        wait_seconds
    )

    final = read_angles(
        robot
    )

    observed_total = (
        final[:3]
        - initial[:3]
    )

    print()
    print(
        "Observed total joint motion:"
    )

    print(
        [
            round(float(x), 3)
            for x in observed_total
        ]
    )

    return final


# ============================================================
# Math
# ============================================================

def damped_pseudoinverse(
    J,
    damping,
):
    """
    J: 2x3.

    J^T (J J^T + λ² I)^-1
    """

    JJt = (
        J
        @ J.T
    )

    regularized = (
        JJt
        + (
            damping ** 2
        )
        * np.eye(2)
    )

    return (
        J.T
        @ np.linalg.inv(
            regularized
        )
    )


def calculate_nullspace_direction(
    J_uv,
    J_area,
):
    """
    Find 1D null-space of J_uv.

    Choose the sign that INCREASES log(area).
    Normalize so max(abs(q)) = 1.
    """

    _, _, Vt = np.linalg.svd(
        J_uv
    )

    n = Vt[-1].copy()

    area_rate = float(
        J_area
        @ n
    )

    if area_rate < 0:
        n = -n
        area_rate = -area_rate

    n = (
        n
        / np.max(
            np.abs(n)
        )
    )

    area_rate_normalized = float(
        J_area
        @ n
    )

    return (
        n,
        area_rate_normalized,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    instruction = parse_instruction(
        args.instruction
    )

    jac_data = load_json(
        JACOBIAN_FILE
    )

    ref_data = load_json(
        STOP_REFERENCE_FILE
    )

    J = np.asarray(
        jac_data["jacobian"],
        dtype=np.float64,
    )

    J_uv = J[:2, :]
    J_area = J[2, :]

    cond_uv = float(
        np.linalg.cond(
            J_uv
        )
    )

    J_uv_pinv = (
        damped_pseudoinverse(
            J_uv,
            args.damping,
        )
    )

    (
        approach_direction,
        expected_area_rate,
    ) = calculate_nullspace_direction(
        J_uv,
        J_area,
    )

    target_u = float(
        ref_data[
            "red_centroid_x"
        ]
    )

    target_v = float(
        ref_data[
            "red_centroid_y"
        ]
    )

    safe_reference_area = float(
        ref_data[
            "target_area"
        ]
    )

    stop_area = (
        safe_reference_area
        * args.area_fraction
    )

    calibration_start = np.asarray(
        jac_data[
            "baseline_angles_deg"
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 72)
    print(
        "LANGUAGE-GUIDED VISUAL SERVO"
    )
    print("=" * 72)

    print()
    print(
        "Instruction:",
        args.instruction,
    )

    print(
        "Parsed target:",
        instruction,
    )

    print()
    print(
        "Mode:",
        (
            "EXECUTE"
            if args.execute
            else "DRY RUN"
        ),
    )

    print()
    print(
        f"Reference (u*,v*) = "
        f"({target_u:.2f}, "
        f"{target_v:.2f})"
    )

    print(
        f"Manual safe area = "
        f"{safe_reference_area:.0f}"
    )

    print(
        f"Automatic first-run stop area = "
        f"{stop_area:.0f} "
        f"({args.area_fraction:.0%})"
    )

    print()
    print(
        f"cond(J_uv) = "
        f"{cond_uv:.3f}"
    )

    print(
        "Approach null-space direction:"
    )

    print(
        [
            round(float(x), 4)
            for x in approach_direction
        ]
    )

    print(
        f"Expected dlog(area)/step-unit = "
        f"{expected_area_rate:+.6f}"
    )

    if cond_uv > 10:
        raise RuntimeError(
            "J_uv is unexpectedly ill-conditioned."
        )

    config = (
        MyCobot280RobotConfig(
            id="language_visual_servo",
            port="auto",
            baudrate=1_000_000,

            # We bypass send_action().
            enable_motion=False,
        )
    )

    robot = MyCobot280Robot(
        config
    )

    session_log = []

    try:
        robot.connect(
            calibrate=False
        )

        if robot.mc is None:
            raise RuntimeError(
                "Controller unavailable."
            )

        current_angles = (
            read_angles(robot)
        )


        print()
        print(
            "Current robot state:"
        )

        print(
            [
                round(float(x), 3)
                for x in current_angles
            ]
        )

        print()
        print(
            "Jacobian calibration state:"
        )

        print(
            [
                round(float(x), 3)
                for x in calibration_start
            ]
        )

        start_difference = (
            current_angles[:3]
            - calibration_start[:3]
        )

        print(
            "J1-J3 offset from calibration:"
        )

        print(
            [
                round(float(x), 3)
                for x in start_difference
            ]
        )

        max_pose_offset = float(
            np.max(
                np.abs(
                    start_difference
                )
            )
        )

        if max_pose_offset > args.start_pose_tolerance:
            print()
            print(
                "CALIBRATION POSE WARNING:"
            )

            print(
                f"Maximum J1-J3 offset = "
                f"{max_pose_offset:.2f} deg"
            )

            if args.execute:
                raise RuntimeError(
                    "START_POSE_TOO_FAR_FROM_CALIBRATION. "
                    "Recalibrate the visual Jacobian "
                    "around the current starting pose."
                )
            else:
                print(
                    "DRY RUN INVALID FOR CONTROL VALIDATION."
                )

                print(
                    "Recalibrate the visual Jacobian "
                    "before execution."
                )

                return

        execution_start = (
            current_angles.copy()
        )

        previous_area = None
        area_decrease_count = 0

        previous_center_norm = None
        center_worse_count = 0

        target_lost_count = 0

        for step_index in range(
            1,
            args.max_steps + 1,
        ):
            print()
            print("=" * 72)
            print(
                f"STEP "
                f"{step_index}/"
                f"{args.max_steps}"
            )
            print("=" * 72)

            frame = read_camera(
                robot
            )

            detection = detect_red(
                frame
            )

            if detection is None:
                target_lost_count += 1

                print(
                    "Red target not detected."
                )

                if target_lost_count >= 2:
                    print(
                        "SAFETY STOP: "
                        "target lost twice."
                    )
                    break

                continue

            target_lost_count = 0

            u = float(
                detection["u"]
            )

            v = float(
                detection["v"]
            )

            area = float(
                detection["area"]
            )

            e_uv = np.asarray(
                [
                    target_u - u,
                    target_v - v,
                ],
                dtype=np.float64,
            )

            center_norm = float(
                np.linalg.norm(
                    e_uv
                )
            )

            print(
                f"Target:"
            )

            print(
                f"  u={u:.2f}, "
                f"v={v:.2f}"
            )

            print(
                f"  area={area:.0f}"
            )

            print(
                f"Reference:"
            )

            print(
                f"  u*={target_u:.2f}, "
                f"v*={target_v:.2f}"
            )

            print(
                f"Error:"
            )

            print(
                f"  du={e_uv[0]:+.2f}px"
            )

            print(
                f"  dv={e_uv[1]:+.2f}px"
            )

            print(
                f"  norm={center_norm:.2f}px"
            )

            print(
                f"Progress:"
            )

            print(
                f"  area target="
                f"{stop_area:.0f}"
            )

            print(
                f"  {100*area/stop_area:.1f}%"
            )

            # Save debug image.
            phase_for_image = (
                "CENTER"
                if center_norm
                > args.center_tolerance_px
                else "APPROACH"
            )

            debug_path = (
                save_debug_frame(
                    frame,
                    detection,
                    target_u,
                    target_v,
                    step_index,
                    phase_for_image,
                )
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if (
                area >= stop_area
                and center_norm
                <= args.approach_center_limit_px
            ):
                print()
                print(
                    "TARGET_REACHED"
                )

                print(
                    "Safe area threshold reached "
                    "while target remains aligned."
                )

                break

            # Hard safety stop if somehow closer
            # than the manually measured safe pose.
            if (
                area
                >= safe_reference_area
                * 1.05
            ):
                print()
                print(
                    "SAFETY STOP:"
                )

                print(
                    "Area exceeded manual "
                    "safe reference."
                )

                break

            # ------------------------------------------------
            # CENTERING CORRECTION
            # ------------------------------------------------

            dq_center = (
                args.center_gain
                * (
                    J_uv_pinv
                    @ e_uv
                )
            )

            # ------------------------------------------------
            # PHASE SELECTION
            # ------------------------------------------------

            if (
                center_norm
                > args.approach_center_limit_px
            ):
                phase = "CENTER"

                # Do not approach while badly misaligned.
                dq = dq_center

            else:
                phase = "APPROACH"

                dq_approach = (
                    approach_direction
                    * args.approach_step_deg
                )

                # Keep correcting image alignment while
                # moving in the visual null-space.
                dq = (
                    dq_center
                    + dq_approach
                )

            # ------------------------------------------------
            # CLIP STEP
            # ------------------------------------------------

            dq = np.clip(
                dq,
                -args.max_step_deg,
                args.max_step_deg,
            )
            # ============================================================
            # Actuator dead-zone compensation
            # ============================================================

            # Empirically determined from the real myCobot.
            # These are COMMAND magnitudes, not expected actual motion.
            MIN_EFFECTIVE_COMMAND = np.asarray(
                [
                    1.50,  # J1
                    1.20,  # J2
                    1.20,  # J3
                ],
                dtype=np.float64,
            )

            # Only apply aggressive dead-zone compensation
            # while still far away from the desired image position.
            #
            # Near the target we do NOT want to force large movements
            # because that could cause oscillation.
            if (
                    phase == "CENTER"
                    and center_norm > 70.0
            ):
                for i in range(3):
                    if (
                            abs(dq[i]) >= 0.15
                            and abs(dq[i])
                            < MIN_EFFECTIVE_COMMAND[i]
                    ):
                        dq[i] = (
                                np.sign(dq[i])
                                * MIN_EFFECTIVE_COMMAND[i]
                        )

                # Respect the global safety bound.
                dq = np.clip(
                    dq,
                    -args.max_step_deg,
                    args.max_step_deg,
                )

            print()
            print(
                "Phase:",
                phase,
            )

            print(
                "Raw command Δq:"
            )

            print(
                [
                    round(float(x), 4)
                    for x in dq
                ]
            )

            # ------------------------------------------------
            # Total travel safety
            # ------------------------------------------------

            current_angles = (
                read_angles(
                    robot
                )
            )

            proposed = (
                current_angles[:3]
                + dq
            )

            total_displacement = (
                proposed
                - execution_start[:3]
            )

            if np.any(
                np.abs(
                    total_displacement
                )
                > args.max_total_joint_travel
            ):
                print()
                print(
                    "SAFETY STOP:"
                )

                print(
                    "Maximum cumulative "
                    "joint travel reached."
                )

                break

            # ------------------------------------------------
            # Trend checks
            # ------------------------------------------------

            if (
                phase == "APPROACH"
                and previous_area
                is not None
            ):
                if (
                    area
                    < previous_area
                    * 0.97
                ):
                    area_decrease_count += 1
                else:
                    area_decrease_count = 0

                if (
                    area_decrease_count
                    >= 3
                ):
                    print()
                    print(
                        "SAFETY STOP:"
                    )

                    print(
                        "Target area decreased "
                        "for three approach steps."
                    )

                    break

            if (
                phase == "CENTER"
                and previous_center_norm
                is not None
            ):
                if (
                    center_norm
                    > previous_center_norm
                    + 15.0
                ):
                    center_worse_count += 1
                else:
                    center_worse_count = 0

                if (
                    center_worse_count
                    >= 3
                ):
                    print()
                    print(
                        "SAFETY STOP:"
                    )

                    print(
                        "Centering error worsened "
                        "for three steps."
                    )

                    break

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            session_log.append(
                {
                    "step": step_index,
                    "phase": phase,
                    "u": u,
                    "v": v,
                    "area": area,
                    "du": float(
                        e_uv[0]
                    ),
                    "dv": float(
                        e_uv[1]
                    ),
                    "center_norm":
                        center_norm,
                    "dq":
                        dq.tolist(),
                    "debug_image":
                        str(debug_path),
                }
            )

            # ------------------------------------------------
            # DRY RUN
            # ------------------------------------------------

            if not args.execute:
                print()
                print(
                    "DRY RUN:"
                )

                print(
                    "No robot motion executed."
                )

                break

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

            actual_angles = (
                execute_delta(
                    robot,
                    dq,
                    args.speed,
                    args.wait,
                )
            )

            print(
                "Actual state after command:"
            )

            print(
                [
                    round(
                        float(x),
                        3,
                    )
                    for x
                    in actual_angles
                ]
            )

            previous_area = area
            previous_center_norm = (
                center_norm
            )

        # ====================================================
        # Save session log
        # ====================================================

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        log_path = (
            OUTPUT_DIR
            / f"servo_{timestamp}.json"
        )

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "instruction":
                        args.instruction,

                    "execute":
                        args.execute,

                    "target_u":
                        target_u,

                    "target_v":
                        target_v,

                    "safe_reference_area":
                        safe_reference_area,

                    "automatic_stop_area":
                        stop_area,

                    "area_fraction":
                        args.area_fraction,

                    "J_uv":
                        J_uv.tolist(),

                    "J_uv_condition":
                        cond_uv,

                    "approach_direction":
                        approach_direction.tolist(),

                    "steps":
                        session_log,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            "Session log:"
        )

        print(
            log_path
        )

    except KeyboardInterrupt:
        print()
        print(
            "CTRL+C received."
        )

        print(
            "Controller stopped."
        )

    finally:
        try:
            robot.disconnect()

        except Exception:
            pass


if __name__ == "__main__":
    main()