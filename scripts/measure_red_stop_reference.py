import json
import sys
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
    / "red_stop_reference"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "red_stop_reference.json"
)


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

        if area < 100:
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
                "contour": contour,
                "color_mode": color_mode,
            }
        )

    if not candidates:
        raise RuntimeError(
            "Red object was not detected."
        )

    return max(
        candidates,
        key=lambda item: item["area"],
    )


def save_result_image(
    frame,
    detection,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if detection["color_mode"] == "RGB":
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

    contour = detection["contour"]

    cv2.drawContours(
        image,
        [contour],
        -1,
        (0, 255, 0),
        2,
    )

    cv2.circle(
        image,
        (cx, cy),
        8,
        (0, 255, 0),
        2,
    )

    text = (
        f"u={cx}, "
        f"v={cy}, "
        f"area={detection['area']:.0f}"
    )

    cv2.putText(
        image,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_path = (
        OUTPUT_DIR
        / "red_stop_reference.jpg"
    )

    ok = cv2.imwrite(
        str(output_path),
        image,
    )

    if not ok:
        raise RuntimeError(
            "Failed to save reference image."
        )

    mask_path = (
        OUTPUT_DIR
        / "red_stop_reference_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        detection["mask"],
    )

    return (
        output_path,
        mask_path,
    )


def main():
    config = MyCobot280RobotConfig(
        id="red_stop_reference",
        port="auto",
        baudrate=1_000_000,

        # Important:
        # this script must never move the robot.
        enable_motion=False,
    )

    robot = MyCobot280Robot(
        config
    )

    try:
        print()
        print("=" * 70)
        print(
            "RED TARGET SAFE STOP REFERENCE"
        )
        print("=" * 70)

        print()
        print(
            "NO ROBOT MOTION WILL BE EXECUTED."
        )

        print()
        print(
            "Before running this measurement:"
        )

        print(
            "1. Manually place the end effector "
            "at a safe final distance from the red object."
        )

        print(
            "2. Lock the robot."
        )

        print(
            "3. Keep the red object stationary."
        )

        print()

        robot.connect(
            calibrate=False
        )

        if robot.mc is None:
            raise RuntimeError(
                "Robot controller unavailable."
            )

        angles = robot.mc.get_angles()

        print(
            "Joint state:"
        )

        print(
            [
                round(
                    float(x),
                    3,
                )
                for x in angles
            ]
        )

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
            frame
        )

        cx = detection["cx"]
        cy = detection["cy"]
        area = detection["area"]

        log_area = float(
            np.log(
                max(
                    area,
                    1.0,
                )
            )
        )

        image_height = int(
            frame.shape[0]
        )

        image_width = int(
            frame.shape[1]
        )

        center_x = (
            image_width / 2.0
        )

        center_y = (
            image_height / 2.0
        )

        error_x = (
            cx - center_x
        )

        error_y = (
            cy - center_y
        )

        image_path, mask_path = (
            save_result_image(
                frame,
                detection,
            )
        )

        print()
        print("=" * 70)
        print(
            "MEASUREMENT RESULT"
        )
        print("=" * 70)

        print()
        print(
            f"Image size: "
            f"{image_width} x {image_height}"
        )

        print(
            f"Image center: "
            f"({center_x:.2f}, "
            f"{center_y:.2f})"
        )

        print()
        print(
            f"Red centroid:"
        )

        print(
            f"u = {cx:.2f}px"
        )

        print(
            f"v = {cy:.2f}px"
        )

        print()
        print(
            f"Center error:"
        )

        print(
            f"du = {error_x:+.2f}px"
        )

        print(
            f"dv = {error_y:+.2f}px"
        )

        print()
        print(
            f"Red area = "
            f"{area:.2f}px"
        )

        print(
            f"log(area) = "
            f"{log_area:.6f}"
        )

        report = {
            "timestamp": (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S"
                )
            ),

            "joint_angles_deg": [
                float(x)
                for x in angles
            ],

            "image_width":
                image_width,

            "image_height":
                image_height,

            "image_center_x":
                center_x,

            "image_center_y":
                center_y,

            "red_centroid_x":
                float(cx),

            "red_centroid_y":
                float(cy),

            "center_error_x":
                float(error_x),

            "center_error_y":
                float(error_y),

            "target_area":
                float(area),

            "target_log_area":
                log_area,

            "reference_image":
                str(image_path),

            "reference_mask":
                str(mask_path),

            "robot_motion_executed":
                False,
        }

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
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
            "Reference image:"
        )

        print(
            image_path
        )

        print()
        print(
            "Report:"
        )

        print(
            REPORT_PATH
        )

        print()
        print("=" * 70)
        print(
            "RED_STOP_REFERENCE_OK"
        )
        print("=" * 70)

    finally:
        try:
            robot.disconnect()

        except Exception:
            pass


if __name__ == "__main__":
    main()