import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open


REPO_ID = "lerobot/smolvla_base"

# LeRobot issue #4415 指出的 pipeline migration 前 revision
LEGACY_REVISION = "3326b100334f"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "smolvla_stats"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "legacy_smolvla_normalization_stats.json"
)


def tensor_to_list(tensor):
    return [
        float(x)
        for x in tensor.detach().float().cpu().flatten()
    ]


def is_interesting_key(key: str) -> bool:
    if not (
        key.endswith(".mean")
        or key.endswith(".std")
    ):
        return False

    return (
        key.startswith("normalize_inputs.")
        or key.startswith("unnormalize_outputs.")
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("RECOVER LEGACY SmolVLA NORMALIZATION STATS")
    print("=" * 70)

    print("Repository:", REPO_ID)
    print("Revision:", LEGACY_REVISION)

    print()
    print(
        "Resolving legacy model.safetensors..."
    )

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.safetensors",
        revision=LEGACY_REVISION,
    )

    print()
    print("Checkpoint:")
    print(model_path)

    result = {
        "repo_id": REPO_ID,
        "revision": LEGACY_REVISION,
        "stats": {},
    }

    print()
    print("=" * 70)
    print("NORMALIZATION / UNNORMALIZATION KEYS")
    print("=" * 70)

    with safe_open(
        model_path,
        framework="pt",
        device="cpu",
    ) as f:

        keys = sorted(f.keys())

        interesting_keys = [
            key
            for key in keys
            if is_interesting_key(key)
        ]

        print()
        print(
            f"Found {len(interesting_keys)} "
            f"normalization tensors."
        )

        for key in interesting_keys:
            tensor = f.get_tensor(key)

            values = tensor_to_list(
                tensor
            )

            result["stats"][key] = {
                "shape": list(tensor.shape),
                "values": values,
            }

            print()
            print("KEY:")
            print(key)

            print(
                "shape:",
                list(tensor.shape),
            )

            print(
                "values:",
                [
                    round(x, 6)
                    for x in values
                ],
            )

    # --------------------------------------------------------
    # Explicitly show the keys we care about most
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SO100 STATE / ACTION CANDIDATES")
    print("=" * 70)

    candidate_fragments = [
        "so100_buffer_observation_state",
        "so100_blue_buffer_observation_state",
        "so100_red_buffer_observation_state",

        "so100_buffer_action",
        "so100_blue_buffer_action",
        "so100_red_buffer_action",
    ]

    for fragment in candidate_fragments:
        matches = [
            key
            for key in result["stats"]
            if fragment in key
        ]

        print()
        print(fragment)

        if not matches:
            print("  <NOT FOUND>")
            continue

        for key in matches:
            print(
                " ",
                key,
            )

            print(
                "   ",
                [
                    round(x, 6)
                    for x in result[
                        "stats"
                    ][key]["values"]
                ],
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "No robot was connected."
    )

    print(
        "No physical action was executed."
    )


if __name__ == "__main__":
    main()