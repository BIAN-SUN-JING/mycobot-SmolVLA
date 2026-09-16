import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lerobot.policies.factory import (
    make_pre_post_processors,
)

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)

from lerobot.policies.utils import (
    build_inference_frame,
)

from lerobot.utils.feature_utils import (
    hw_to_dataset_features,
)

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

LEGACY_STATS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "smolvla_stats"
    / "legacy_smolvla_normalization_stats.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "zero_shot_fixed_stats"
)


SO100_STATE_MEAN_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.mean"
)

SO100_STATE_STD_KEY = (
    "normalize_inputs."
    "so100_buffer_observation_state.std"
)

SO100_ACTION_MEAN_KEY = (
    "unnormalize_outputs."
    "so100_buffer_action.mean"
)

SO100_ACTION_STD_KEY = (
    "unnormalize_outputs."
    "so100_buffer_action.std"
)


SO100_ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


MYCOBOT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
]


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SmolVLA with manually restored "
            "legacy SO100 normalization stats."
        )
    )

    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="fixed_stats_run",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_legacy_stats():
    if not LEGACY_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Legacy stats file not found:\n"
            f"{LEGACY_STATS_FILE}"
        )

    with LEGACY_STATS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    stats = data["stats"]

    required = [
        SO100_STATE_MEAN_KEY,
        SO100_STATE_STD_KEY,
        SO100_ACTION_MEAN_KEY,
        SO100_ACTION_STD_KEY,
    ]

    for key in required:
        if key not in stats:
            raise KeyError(
                f"Required legacy stats key missing:\n"
                f"{key}"
            )

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

    action_mean = torch.tensor(
        stats[
            SO100_ACTION_MEAN_KEY
        ]["values"],
        dtype=torch.float32,
    )

    action_std = torch.tensor(
        stats[
            SO100_ACTION_STD_KEY
        ]["values"],
        dtype=torch.float32,
    )

    return (
        state_mean,
        state_std,
        action_mean,
        action_std,
    )


def tensor_list(tensor):
    return (
        tensor
        .detach()
        .float()
        .cpu()
        .flatten()
        .tolist()
    )


def print_vector(
    title,
    values,
    decimals=4,
):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        [
            round(float(x), decimals)
            for x in values
        ]
    )


def save_result(
    label,
    task,
    seed,
    current_state,
    normalized_state,
    normalized_action,
    so100_action,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_label = "".join(
        char
        if char.isalnum() or char in "-_"
        else "_"
        for char in label
    )

    path = (
        OUTPUT_DIR
        / f"{timestamp}_{safe_label}.json"
    )

    data = {
        "timestamp": timestamp,
        "model": MODEL_ID,
        "task": task,
        "seed": seed,

        "mycobot_state_degrees":
            current_state,

        "manual_so100_normalized_state":
            normalized_state,

        "model_normalized_action":
            normalized_action,

        "manual_so100_unnormalized_action":
            so100_action,

        "so100_action_names":
            SO100_ACTION_NAMES,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


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
    print("=" * 70)
    print("SmolVLA ZERO-SHOT TEST")
    print("LEGACY SO100 STATS RESTORED")
    print("=" * 70)

    print("Device :", device)
    print("Model  :", MODEL_ID)
    print("Task   :", task)
    print("Seed   :", seed)
    print("Label  :", label)

    # ========================================================
    # Load legacy SO100 statistics
    # ========================================================

    print()
    print("[1/8] Loading legacy SO100 stats...")

    (
        state_mean,
        state_std,
        action_mean,
        action_std,
    ) = load_legacy_stats()

    print_vector(
        "SO100 STATE MEAN",
        state_mean,
    )

    print_vector(
        "SO100 STATE STD",
        state_std,
    )

    print_vector(
        "SO100 ACTION MEAN",
        action_mean,
    )

    print_vector(
        "SO100 ACTION STD",
        action_std,
    )

    print()
    print("LEGACY_STATS_OK")

    # ========================================================
    # Load pretrained SmolVLA
    # ========================================================

    print()
    print("[2/8] Loading SmolVLA...")

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            MODEL_ID
        )
        .to(device)
        .eval()
    )

    print("SMOLVLA_LOAD_OK")

    # ========================================================
    # Create standard LeRobot preprocessor
    #
    # We still use it for:
    # - image processing
    # - language tokenization
    # - batching
    # - device transfer
    #
    # But we MANUALLY overwrite observation.state afterwards.
    # ========================================================

    print()
    print("[3/8] Creating processor stack...")

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

    print("PROCESSOR_STACK_OK")

    # ========================================================
    # Robot
    # ========================================================

    robot_config = MyCobot280RobotConfig(
        id="mycobot280_fixed_stats",
        port="auto",
        baudrate=1_000_000,

        # CRITICAL:
        # no physical motion allowed.
        enable_motion=False,
    )

    robot = MyCobot280Robot(
        robot_config
    )

    try:

        # ====================================================
        # Connect
        # ====================================================

        print()
        print(
            "[4/8] Connecting robot and camera..."
        )

        robot.connect(
            calibrate=False
        )

        print("ROBOT_CONNECTED")

        # ====================================================
        # Hardware features
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

        # ====================================================
        # Observation
        # ====================================================

        print()
        print(
            "[5/8] Reading real observation..."
        )

        observation = (
            robot.get_observation()
        )

        current_state = [
            float(
                observation[name]
            )
            for name in MYCOBOT_NAMES
        ]

        print_vector(
            "CURRENT MYCOBOT STATE (degrees)",
            current_state,
            decimals=3,
        )

        print()
        print(
            "Camera shape:",
            observation[
                "camera1"
            ].shape,
        )

        # ====================================================
        # Standard inference frame
        # ====================================================

        inference_frame = (
            build_inference_frame(
                observation=observation,
                ds_features=dataset_features,
                device=device,
                task=task,
                robot_type="mycobot280",
            )
        )

        batch = preprocessor(
            inference_frame
        )

        print()
        print("BASE_PREPROCESS_OK")

        # ====================================================
        # MANUAL LEGACY STATE NORMALIZATION
        # ====================================================

        print()
        print(
            "[6/8] Applying legacy "
            "SO100 state normalization..."
        )

        raw_state_tensor = (
            batch[
                "observation.state"
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        state_mean_device = (
            state_mean
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        state_std_device = (
            state_std
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        normalized_state_tensor = (
            (
                raw_state_tensor
                - state_mean_device
            )
            / state_std_device
        )

        # Overwrite broken/un-normalized pipeline result.
        batch[
            "observation.state"
        ] = normalized_state_tensor

        normalized_state = (
            tensor_list(
                normalized_state_tensor
            )
        )

        print_vector(
            "MANUAL SO100-NORMALIZED INPUT STATE",
            normalized_state,
            decimals=4,
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This fixes the numerical normalization "
            "expected by the pretrained SO100 policy."
        )

        print(
            "It does NOT make myCobot joints physically "
            "equivalent to SO100 joints."
        )

        # ====================================================
        # Deterministic inference
        # ====================================================

        if hasattr(
            policy,
            "reset",
        ):
            policy.reset()

        set_seed(
            seed
        )

        print()
        print(
            "[7/8] Running SmolVLA inference..."
        )

        with torch.no_grad():
            normalized_action_tensor = (
                policy.select_action(
                    batch
                )
            )

        print("MODEL_FORWARD_OK")

        normalized_action = (
            tensor_list(
                normalized_action_tensor
            )[:6]
        )

        print_vector(
            "MODEL NORMALIZED ACTION",
            normalized_action,
            decimals=4,
        )

        # ====================================================
        # MANUAL LEGACY ACTION UNNORMALIZATION
        # ====================================================

        action_mean_device = (
            action_mean
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        action_std_device = (
            action_std
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        source_action_tensor = (
            normalized_action_tensor
            * action_std_device
            + action_mean_device
        )

        source_action = (
            tensor_list(
                source_action_tensor
            )[:6]
        )

        print_vector(
            "MANUAL SO100-UNNORMALIZED ACTION",
            source_action,
            decimals=4,
        )

        print()
        print("=" * 70)
        print("SO100 SOURCE ACTION SEMANTICS")
        print("=" * 70)

        for name, value in zip(
            SO100_ACTION_NAMES,
            source_action,
        ):
            print(
                f"{name:16s}: "
                f"{value:9.4f}"
            )

        print()
        print(
            "WARNING:"
        )

        print(
            "These are SO100 source-embodiment values."
        )

        print(
            "They are NOT myCobot J1-J6 targets."
        )

        print()
        print(
            "In particular:"
        )

        print(
            "SO100 dimension 6 = gripper"
        )

        print(
            "myCobot dimension 6 = wrist joint J6"
        )

        # ====================================================
        # Save
        # ====================================================

        print()
        print(
            "[8/8] Saving diagnostic result..."
        )

        result_path = save_result(
            label=label,
            task=task,
            seed=seed,
            current_state=current_state,
            normalized_state=normalized_state,
            normalized_action=normalized_action,
            so100_action=source_action,
        )

        print()
        print(
            "Saved to:"
        )

        print(
            result_path
        )

        print()
        print("=" * 70)
        print("FIXED_STATS_ZERO_SHOT_OK")
        print("=" * 70)

        print()
        print(
            "NO ACTION WAS SENT TO THE ROBOT."
        )

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()