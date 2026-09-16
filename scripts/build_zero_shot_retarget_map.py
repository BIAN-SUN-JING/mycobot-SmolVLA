import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "zero_shot_fixed_stats"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "retarget"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "zero_shot_retarget_report.json"
)


SOURCE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


MYCOBOT_NAMES = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
]


GROUP_PATTERNS = {
    "right_toward":
        "*clean_right_toward_*.json",

    "left_toward":
        "*clean_left_toward_*.json",

    "right_away":
        "*clean_right_away_*.json",
}


EPS = 1e-8


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_group(pattern):
    files = sorted(
        DATA_DIR.glob(pattern)
    )

    if len(files) < 3:
        raise RuntimeError(
            f"Need at least 3 files for "
            f"pattern:\n{pattern}\n"
            f"Found: {len(files)}"
        )

    records = []

    for path in files:
        data = load_json(
            path
        )

        action = np.asarray(
            data[
                "model_normalized_action"
            ],
            dtype=np.float64,
        )

        state = np.asarray(
            data[
                "mycobot_state_degrees"
            ],
            dtype=np.float64,
        )

        records.append(
            {
                "path": str(path),
                "action": action,
                "state": state,
            }
        )

    return records


def stack_actions(records):
    return np.stack(
        [
            item["action"]
            for item in records
        ],
        axis=0,
    )


def stack_states(records):
    return np.stack(
        [
            item["state"]
            for item in records
        ],
        axis=0,
    )


def l2(x):
    return float(
        np.linalg.norm(x)
    )


def vector_list(x):
    return [
        float(v)
        for v in x
    ]


def group_summary(records):
    actions = stack_actions(
        records
    )

    states = stack_states(
        records
    )

    action_mean = actions.mean(
        axis=0
    )

    action_std = actions.std(
        axis=0,
        ddof=0,
    )

    state_mean = states.mean(
        axis=0
    )

    state_range = (
        states.max(axis=0)
        - states.min(axis=0)
    )

    distances = np.linalg.norm(
        actions - action_mean,
        axis=1,
    )

    return {
        "action_mean":
            vector_list(action_mean),

        "action_std":
            vector_list(action_std),

        "mean_l2_to_centroid":
            float(distances.mean()),

        "max_l2_to_centroid":
            float(distances.max()),

        "state_mean":
            vector_list(state_mean),

        "state_range_deg":
            vector_list(state_range),

        "num_samples":
            int(actions.shape[0]),
    }


def make_candidates(
    visual_snr,
):
    # Ignore dimension 6:
    # SO100 dim6 = gripper.
    arm_indices = list(
        range(5)
    )

    ranked = sorted(
        arm_indices,
        key=lambda i:
            visual_snr[i],
        reverse=True,
    )

    top1 = ranked[:1]
    top2 = ranked[:2]
    top3 = ranked[:3]

    candidates = {}

    # --------------------------------------------------------
    # Candidate A:
    # semantic same-index heuristic.
    # This is NOT kinematically verified.
    # --------------------------------------------------------

    candidates[
        "semantic_5d"
    ] = {
        "description": (
            "Heuristic same-index mapping: "
            "SO100 arm dims 1-5 -> "
            "myCobot J1-J5; J6 held."
        ),

        "source_to_target": [
            {
                "source_dim": 1,
                "source_name":
                    SOURCE_NAMES[0],
                "target_joint": "J1",
            },
            {
                "source_dim": 2,
                "source_name":
                    SOURCE_NAMES[1],
                "target_joint": "J2",
            },
            {
                "source_dim": 3,
                "source_name":
                    SOURCE_NAMES[2],
                "target_joint": "J3",
            },
            {
                "source_dim": 4,
                "source_name":
                    SOURCE_NAMES[3],
                "target_joint": "J4",
            },
            {
                "source_dim": 5,
                "source_name":
                    SOURCE_NAMES[4],
                "target_joint": "J5",
            },
        ],

        "hold_joints": [
            "J6",
        ],

        "verified": False,
    }

    # --------------------------------------------------------
    # Candidate B:
    # Only use source dimensions with highest visual SNR,
    # while keeping their same-index target joints.
    # --------------------------------------------------------

    def sparse_candidate(
        indices,
        name,
    ):
        mapping = []

        for i in indices:
            mapping.append(
                {
                    "source_dim":
                        i + 1,

                    "source_name":
                        SOURCE_NAMES[i],

                    "target_joint":
                        MYCOBOT_NAMES[i],
                }
            )

        active_targets = {
            item["target_joint"]
            for item in mapping
        }

        hold = [
            joint
            for joint in MYCOBOT_NAMES
            if joint not in active_targets
        ]

        return {
            "description": (
                f"Visual-SNR sparse "
                f"candidate using "
                f"{len(indices)} "
                f"most responsive "
                f"SO100 arm dimensions."
            ),

            "source_to_target":
                mapping,

            "hold_joints":
                hold,

            "verified":
                False,
        }

    candidates[
        "visual_top1"
    ] = sparse_candidate(
        top1,
        "visual_top1",
    )

    candidates[
        "visual_top2"
    ] = sparse_candidate(
        top2,
        "visual_top2",
    )

    candidates[
        "visual_top3"
    ] = sparse_candidate(
        top3,
        "visual_top3",
    )

    return (
        candidates,
        ranked,
    )


def main():
    print()
    print("=" * 72)
    print(
        "ZERO-SHOT RETARGET MAP BUILDER"
    )
    print("=" * 72)

    groups = {}

    for name, pattern in (
        GROUP_PATTERNS.items()
    ):
        records = load_group(
            pattern
        )

        groups[name] = records

        print(
            f"{name}: "
            f"{len(records)} samples"
        )

    rt = stack_actions(
        groups[
            "right_toward"
        ]
    )

    lt = stack_actions(
        groups[
            "left_toward"
        ]
    )

    ra = stack_actions(
        groups[
            "right_away"
        ]
    )

    mu_rt = rt.mean(axis=0)
    mu_lt = lt.mean(axis=0)
    mu_ra = ra.mean(axis=0)

    std_rt = rt.std(
        axis=0,
        ddof=0,
    )

    visual_effect = (
        mu_lt
        - mu_rt
    )

    language_effect = (
        mu_ra
        - mu_rt
    )

    repeat_distances = (
        np.linalg.norm(
            rt - mu_rt,
            axis=1,
        )
    )

    d_noise = float(
        repeat_distances.mean()
    )

    d_visual = l2(
        visual_effect
    )

    d_language = l2(
        language_effect
    )

    snr_visual = (
        d_visual
        / max(
            d_noise,
            EPS,
        )
    )

    snr_language = (
        d_language
        / max(
            d_noise,
            EPS,
        )
    )

    per_dim_visual_snr = (
        np.abs(
            visual_effect
        )
        / (
            std_rt
            + EPS
        )
    )

    per_dim_language_snr = (
        np.abs(
            language_effect
        )
        / (
            std_rt
            + EPS
        )
    )

    print()
    print("=" * 72)
    print("GLOBAL METRICS")
    print("=" * 72)

    print(
        f"D_noise    = "
        f"{d_noise:.6f}"
    )

    print(
        f"D_visual   = "
        f"{d_visual:.6f}"
    )

    print(
        f"D_language = "
        f"{d_language:.6f}"
    )

    print(
        f"SNR_visual = "
        f"{snr_visual:.3f}"
    )

    print(
        f"SNR_lang   = "
        f"{snr_language:.3f}"
    )

    print()
    print("=" * 72)
    print(
        "PER-DIMENSION VISUAL EFFECT"
    )
    print("=" * 72)

    for i in range(6):
        print(
            f"dim{i + 1} "
            f"{SOURCE_NAMES[i]:16s} "
            f"effect="
            f"{visual_effect[i]:+9.5f} "
            f"noise_std="
            f"{std_rt[i]:8.5f} "
            f"SNR="
            f"{per_dim_visual_snr[i]:7.3f}"
        )

    print()
    print("=" * 72)
    print(
        "PER-DIMENSION LANGUAGE EFFECT"
    )
    print("=" * 72)

    for i in range(6):
        print(
            f"dim{i + 1} "
            f"{SOURCE_NAMES[i]:16s} "
            f"effect="
            f"{language_effect[i]:+9.5f} "
            f"noise_std="
            f"{std_rt[i]:8.5f} "
            f"SNR="
            f"{per_dim_language_snr[i]:7.3f}"
        )

    candidates, ranked = (
        make_candidates(
            per_dim_visual_snr
        )
    )

    print()
    print("=" * 72)
    print(
        "VISUAL RESPONSIVENESS RANKING"
    )
    print("=" * 72)

    # Only first five are robot arm DOFs.
    for rank, i in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank}. "
            f"dim{i + 1} "
            f"{SOURCE_NAMES[i]} "
            f"SNR="
            f"{per_dim_visual_snr[i]:.3f}"
        )

    print()
    print("=" * 72)
    print("CANDIDATE MAPS")
    print("=" * 72)

    for name, candidate in (
        candidates.items()
    ):
        print()
        print(name)

        print(
            " ",
            candidate[
                "description"
            ],
        )

        for item in candidate[
            "source_to_target"
        ]:
            print(
                f"   source dim "
                f"{item['source_dim']} "
                f"({item['source_name']}) "
                f"-> "
                f"{item['target_joint']}"
            )

        print(
            "   hold:",
            candidate[
                "hold_joints"
            ],
        )

    report = {
        "groups": {
            name:
                group_summary(
                    records
                )
            for name, records
            in groups.items()
        },

        "metrics": {
            "d_noise":
                d_noise,

            "d_visual":
                d_visual,

            "d_language":
                d_language,

            "snr_visual":
                snr_visual,

            "snr_language":
                snr_language,

            "right_toward_mean":
                vector_list(
                    mu_rt
                ),

            "left_toward_mean":
                vector_list(
                    mu_lt
                ),

            "right_away_mean":
                vector_list(
                    mu_ra
                ),

            "visual_effect":
                vector_list(
                    visual_effect
                ),

            "language_effect":
                vector_list(
                    language_effect
                ),

            "right_toward_std":
                vector_list(
                    std_rt
                ),

            "per_dim_visual_snr":
                vector_list(
                    per_dim_visual_snr
                ),

            "per_dim_language_snr":
                vector_list(
                    per_dim_language_snr
                ),
        },

        "source_semantics":
            SOURCE_NAMES,

        "visual_arm_rank":
            [
                int(i + 1)
                for i in ranked
            ],

        "candidates":
            candidates,

        "notes": [
            (
                "SO100 dimension 6 is "
                "gripper and is excluded "
                "from arm retargeting."
            ),
            (
                "Candidate maps are "
                "heuristics only."
            ),
            (
                "Joint signs are not "
                "physically calibrated."
            ),
            (
                "No candidate is approved "
                "for robot execution."
            ),
        ],
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
    print("=" * 72)
    print("REPORT SAVED")
    print("=" * 72)

    print(
        REPORT_PATH
    )

    print()
    print(
        "NO ROBOT ACTION WAS EXECUTED."
    )


if __name__ == "__main__":
    main()