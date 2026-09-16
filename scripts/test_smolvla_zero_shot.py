import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LeRobot / SmolVLA
# ============================================================

from lerobot.policies.factory import (
    make_pre_post_processors,
)

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)

from lerobot.policies.utils import (
    build_inference_frame,
    make_robot_action,
)

from lerobot.utils.feature_utils import (
    hw_to_dataset_features,
)


# ============================================================
# Local myCobot adapter
# ============================================================

from hardware.mycobot_robot import (
    MyCobot280Robot,
    MyCobot280RobotConfig,
)


# ============================================================
# Constants
# ============================================================

MODEL_ID = "lerobot/smolvla_base"

DEFAULT_TASK = (
    "Move the end effector toward the red object."
)

DEFAULT_SEED = 42

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "zero_shot_runs"
)


# ============================================================
# Utility
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained SmolVLA zero-shot inference "
            "on a real myCobot 280 without executing actions."
        )
    )

    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Natural-language robot instruction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for SmolVLA action sampling.",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="run",
        help=(
            "Name used when saving the result JSON. "
            "Example: left_toward"
        ),
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve repeatability on CUDA.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_to_list(
    tensor: torch.Tensor,
    count: int | None = None,
):
    values = (
        tensor
        .detach()
        .float()
        .cpu()
        .flatten()
        .tolist()
    )

    if count is not None:
        values = values[:count]

    return [
        float(x)
        for x in values
    ]


def print_vector(
    title: str,
    values,
    decimals: int = 4,
):
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)

    print(
        [
            round(float(x), decimals)
            for x in values
        ]
    )


def save_result(
    label: str,
    task: str,
    seed: int,
    current_state,
    normalized_state,
    raw_action,
    postprocessed_action,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_label = "".join(
        c if c.isalnum() or c in "-_"
        else "_"
        for c in label
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"{timestamp}_{safe_label}.json"
    )

    result = {
        "timestamp": timestamp,
        "model": MODEL_ID,
        "task": task,
        "seed": seed,
        "current_state_deg": current_state,
        "normalized_input_state": normalized_state,
        "raw_policy_action": raw_action,
        "postprocessed_action": postprocessed_action,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    task = args.task
    seed = args.seed
    label = args.label

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("SmolVLA PRETRAINED ZERO-SHOT TEST")
    print("=" * 60)

    print(
        f"Device : {device}"
    )

    print(
        f"Model  : {MODEL_ID}"
    )

    print(
        f"Task   : {task}"
    )

    print(
        f"Seed   : {seed}"
    )

    print(
        f"Label  : {label}"
    )

    # ========================================================
    # Load pretrained policy
    # ========================================================

    print()
    print(
        "[1/7] Loading pretrained SmolVLA..."
    )

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            MODEL_ID
        )
        .to(device)
        .eval()
    )

    print(
        "SMOLVLA_LOAD_OK"
    )

    # ========================================================
    # Build preprocessor / postprocessor
    # ========================================================

    print()
    print(
        "[2/7] Creating preprocessors..."
    )

    preprocessor, postprocessor = (
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

    print(
        "PROCESSOR_STACK_OK"
    )

    # ========================================================
    # Configure robot
    #
    # IMPORTANT:
    # enable_motion=False ensures this script can never execute
    # policy output on the physical robot.
    # ========================================================

    robot_config = MyCobot280RobotConfig(
        id="mycobot280_main",
        port="auto",
        baudrate=1_000_000,

        # Safety:
        enable_motion=False,
    )

    robot = MyCobot280Robot(
        robot_config
    )

    try:

        # ====================================================
        # Connect real robot + camera
        # ====================================================

        print()
        print(
            "[3/7] Connecting myCobot and camera..."
        )

        robot.connect(
            calibrate=False
        )

        print(
            "ROBOT_CONNECTED"
        )

        # ====================================================
        # Convert hardware features
        # ====================================================

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

        dataset_features = {
            **action_features,
            **observation_features,
        }

        print()
        print("=" * 50)
        print("DATASET FEATURES")
        print("=" * 50)

        for key, value in dataset_features.items():
            print(
                f"{key}:"
            )

            print(
                f"    {value}"
            )

        # ====================================================
        # Read real observation
        # ====================================================

        print()
        print(
            "[4/7] Reading real robot observation..."
        )

        observation = (
            robot.get_observation()
        )

        current_state = []

        for i in range(
            1,
            7,
        ):
            key = (
                f"joint_{i}.pos"
            )

            value = float(
                observation[key]
            )

            current_state.append(
                value
            )

        print_vector(
            "CURRENT MYCOBOT JOINT STATE (degrees)",
            current_state,
            decimals=3,
        )

        image = observation[
            "camera1"
        ]

        print()
        print(
            "Camera shape:",
            image.shape,
        )

        print(
            "Camera dtype:",
            image.dtype,
        )

        # ====================================================
        # Build standard LeRobot inference frame
        # ====================================================

        print()
        print(
            "[5/7] Building inference frame..."
        )

        inference_frame = (
            build_inference_frame(
                observation=observation,
                ds_features=dataset_features,
                device=device,
                task=task,
                robot_type="mycobot280",
            )
        )

        print()
        print(
            "Inference frame keys:"
        )

        for key in inference_frame:
            print(
                f"  {key}"
            )

        # ====================================================
        # Preprocess
        # ====================================================

        batch = preprocessor(
            inference_frame
        )

        print()
        print(
            "PREPROCESS_OK"
        )

        # ====================================================
        # Inspect normalized robot state
        # ====================================================

        if (
            "observation.state"
            not in batch
        ):
            raise KeyError(
                "Preprocessed batch does not contain "
                "'observation.state'."
            )

        normalized_state = (
            tensor_to_list(
                batch[
                    "observation.state"
                ],
                count=6,
            )
        )

        print_vector(
            "NORMALIZED INPUT STATE",
            normalized_state,
            decimals=4,
        )

        print()
        print(
            "NOTE:"
        )

        print(
            "These values are produced using the "
            "pretrained checkpoint's normalization "
            "statistics, not myCobot-specific statistics."
        )

        # ====================================================
        # Reset action queue
        # ====================================================

        if hasattr(
            policy,
            "reset",
        ):
            policy.reset()

            print()
            print(
                "POLICY_RESET_OK"
            )

        # ====================================================
        # Deterministic seed
        # ====================================================

        set_seed(
            seed
        )

        # ====================================================
        # Zero-shot inference
        # ====================================================

        print()
        print(
            "[6/7] Running SmolVLA inference..."
        )

        with torch.no_grad():
            action_tensor = (
                policy.select_action(
                    batch
                )
            )

        print(
            "MODEL_FORWARD_OK"
        )

        # ====================================================
        # Raw action BEFORE checkpoint postprocessor
        # ====================================================

        raw_action = (
            tensor_to_list(
                action_tensor,
                count=6,
            )
        )

        print_vector(
            "RAW POLICY ACTION "
            "(before postprocessor)",
            raw_action,
            decimals=4,
        )

        # ====================================================
        # Apply checkpoint postprocessor
        # ====================================================

        processed_action_tensor = (
            postprocessor(
                action_tensor
            )
        )

        postprocessed_action_vector = (
            tensor_to_list(
                processed_action_tensor,
                count=6,
            )
        )

        print_vector(
            "POSTPROCESSED ACTION VECTOR",
            postprocessed_action_vector,
            decimals=4,
        )

        # ====================================================
        # Convert to robot-action dictionary
        #
        # IMPORTANT:
        # This is only a numerical conversion.
        # It does NOT mean the output is physically compatible
        # with myCobot.
        # ====================================================

        robot_action = (
            make_robot_action(
                processed_action_tensor,
                dataset_features,
            )
        )

        print()
        print("=" * 50)
        print("ZERO-SHOT NAMED ACTION")
        print("=" * 50)

        named_action_values = []

        for i in range(
            1,
            7,
        ):
            key = (
                f"joint_{i}.pos"
            )

            value = float(
                robot_action[key]
            )

            named_action_values.append(
                value
            )

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        # ====================================================
        # Compare current robot state against output
        # ====================================================

        print()
        print("=" * 70)
        print("CURRENT MYCOBOT STATE vs PRETRAINED POLICY OUTPUT")
        print("=" * 70)

        for i in range(
            6
        ):
            current = (
                current_state[i]
            )

            predicted = (
                named_action_values[i]
            )

            delta = (
                predicted
                - current
            )

            print(
                f"J{i + 1}: "
                f"current={current:9.3f}  "
                f"predicted={predicted:9.3f}  "
                f"numeric_delta={delta:9.3f}"
            )

        print()
        print(
            "WARNING:"
        )

        print(
            "numeric_delta above is diagnostic only."
        )

        print(
            "The pretrained output must NOT be interpreted "
            "as myCobot joint-degree targets."
        )

        # ====================================================
        # Save result
        # ====================================================

        print()
        print(
            "[7/7] Saving diagnostic result..."
        )

        result_path = save_result(
            label=label,
            task=task,
            seed=seed,
            current_state=current_state,
            normalized_state=normalized_state,
            raw_action=raw_action,
            postprocessed_action=named_action_values,
        )

        print(
            "Result saved to:"
        )

        print(
            result_path
        )

        # ====================================================
        # Final safety statement
        # ====================================================

        print()
        print("=" * 60)
        print("ZERO_SHOT_INFERENCE_OK")
        print("=" * 60)

        print()
        print(
            "NO ACTION WAS SENT TO THE ROBOT."
        )

        print(
            "Robot motion remains disabled."
        )

    finally:

        try:
            robot.disconnect()
        except Exception as exc:
            print(
                f"[Warning] Disconnect failed: {exc}"
            )


if __name__ == "__main__":
    main()