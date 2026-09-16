import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lerobot.policies.factory import (
    make_pre_post_processors,
)

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)


DEFAULT_MODEL = "lerobot/smolvla_base"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "smolvla_stats"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect SmolVLA pretrained normalization and "
            "unnormalization statistics."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def tensor_summary(
    tensor: torch.Tensor,
):
    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
    )

    flat = tensor.flatten()

    values = [
        round(float(x), 6)
        for x in flat[:20]
    ]

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": (
            float(flat.min())
            if flat.numel() > 0
            else None
        ),
        "max": (
            float(flat.max())
            if flat.numel() > 0
            else None
        ),
        "mean": (
            float(flat.mean())
            if flat.numel() > 0
            else None
        ),
        "values_first_20": values,
    }


def print_tensor(
    key: str,
    tensor: torch.Tensor,
):
    summary = tensor_summary(
        tensor
    )

    print()
    print(
        f"KEY: {key}"
    )

    print(
        "  shape:",
        summary["shape"],
    )

    print(
        "  dtype:",
        summary["dtype"],
    )

    print(
        "  min:",
        summary["min"],
    )

    print(
        "  max:",
        summary["max"],
    )

    print(
        "  mean:",
        summary["mean"],
    )

    print(
        "  first values:",
        summary["values_first_20"],
    )


def download_model_file(
    repo_id: str,
    filename: str,
    revision: str | None,
):
    print(
        f"[Hub] Resolving {filename}..."
    )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )

    print(
        f"[Hub] {filename}:"
    )

    print(
        f"      {path}"
    )

    return Path(path)


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def find_processor_step(
    processor_json: dict,
    registry_fragment: str,
):
    for index, step in enumerate(
        processor_json.get(
            "steps",
            [],
        )
    ):
        registry_name = step.get(
            "registry_name",
            "",
        )

        if (
            registry_fragment
            in registry_name
        ):
            return index, step

    return None, None


def inspect_processor_json(
    title: str,
    data: dict,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "Pipeline name:",
        data.get("name"),
    )

    steps = data.get(
        "steps",
        [],
    )

    for index, step in enumerate(
        steps
    ):
        print()
        print(
            f"Step {index}: "
            f"{step.get('registry_name')}"
        )

        config = step.get(
            "config",
            {},
        )

        if "norm_map" in config:
            print(
                "  norm_map:"
            )

            print(
                json.dumps(
                    config["norm_map"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if "features" in config:
            print(
                "  features:"
            )

            print(
                json.dumps(
                    config["features"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if "state_file" in step:
            print(
                "  state_file:",
                step["state_file"],
            )


def extract_feature_names_from_keys(
    state_dict: dict[str, torch.Tensor],
):
    result = set()

    known_suffixes = (
        ".mean",
        ".std",
        ".min",
        ".max",
        ".q01",
        ".q99",
    )

    for key in state_dict:
        found = False

        for suffix in known_suffixes:
            if key.endswith(suffix):
                result.add(
                    key[
                        :-len(suffix)
                    ]
                )

                found = True
                break

        if not found:
            # Generic fallback:
            # strip the final segment
            if "." in key:
                result.add(
                    key.rsplit(
                        ".",
                        1,
                    )[0]
                )

    return sorted(
        result
    )


def candidate_keys(
    state_dict: dict[str, torch.Tensor],
    feature_fragment: str,
):
    return [
        key
        for key in state_dict
        if feature_fragment in key
    ]


def analyze_expected_keys(
    title: str,
    state_dict: dict[str, torch.Tensor],
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    expected = [
        "observation.state.mean",
        "observation.state.std",
        "action.mean",
        "action.std",
    ]

    print()
    print("Exact expected keys:")

    for key in expected:
        exists = (
            key in state_dict
        )

        marker = (
            "FOUND"
            if exists
            else "MISSING"
        )

        print(
            f"  [{marker}] {key}"
        )

    print()
    print(
        "Candidates containing "
        "'observation.state':"
    )

    obs_candidates = (
        candidate_keys(
            state_dict,
            "observation.state",
        )
    )

    if obs_candidates:
        for key in obs_candidates:
            print(
                f"  {key}"
            )
    else:
        print(
            "  <none>"
        )

    print()
    print(
        "Candidates containing "
        "'action':"
    )

    action_candidates = (
        candidate_keys(
            state_dict,
            "action",
        )
    )

    if action_candidates:
        for key in action_candidates:
            print(
                f"  {key}"
            )
    else:
        print(
            "  <none>"
        )

    print()
    print(
        "Detected feature prefixes:"
    )

    features = (
        extract_feature_names_from_keys(
            state_dict
        )
    )

    for feature in features:
        print(
            f"  {feature}"
        )


def inspect_safetensors(
    title: str,
    path: Path,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "File:",
        path,
    )

    state_dict = load_file(
        str(path)
    )

    print(
        "Number of tensors:",
        len(state_dict),
    )

    print()
    print(
        "ALL SAFETENSOR KEYS:"
    )

    for key in sorted(
        state_dict.keys()
    ):
        print(
            f"  {key}"
        )

    for key in sorted(
        state_dict.keys()
    ):
        print_tensor(
            key,
            state_dict[key],
        )

    return state_dict


def print_object_attributes(
    obj: Any,
):
    interesting = [
        "features",
        "norm_map",
        "stats",
        "_tensor_stats",
        "device",
        "dtype",
        "eps",
    ]

    for attr in interesting:
        if not hasattr(
            obj,
            attr,
        ):
            continue

        value = getattr(
            obj,
            attr,
        )

        print()
        print(
            f"    {attr}:"
        )

        if isinstance(
            value,
            dict,
        ):
            for key, item in value.items():
                print(
                    f"      {key}: "
                    f"{item}"
                )
        else:
            print(
                f"      {value}"
            )


def inspect_runtime_pipeline(
    preprocessor,
    postprocessor,
):
    print()
    print("=" * 70)
    print("RUNTIME PREPROCESSOR")
    print("=" * 70)

    for index, step in enumerate(
        preprocessor.steps
    ):
        print()
        print(
            f"PRE STEP {index}: "
            f"{type(step).__name__}"
        )

        print_object_attributes(
            step
        )

    print()
    print("=" * 70)
    print("RUNTIME POSTPROCESSOR")
    print("=" * 70)

    for index, step in enumerate(
        postprocessor.steps
    ):
        print()
        print(
            f"POST STEP {index}: "
            f"{type(step).__name__}"
        )

        print_object_attributes(
            step
        )


def serialize_state_dict(
    state_dict: dict[str, torch.Tensor],
):
    result = {}

    for key, tensor in state_dict.items():
        result[key] = (
            tensor_summary(
                tensor
            )
        )

    return result


def main():
    args = parse_args()

    repo_id = args.model
    revision = args.revision
    device = args.device

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("SmolVLA NORMALIZATION STATS INSPECTOR")
    print("=" * 70)

    print(
        "Model:",
        repo_id,
    )

    print(
        "Revision:",
        revision,
    )

    print(
        "Device:",
        device,
    )

    # ========================================================
    # Download / resolve metadata files
    # ========================================================

    pre_json_path = (
        download_model_file(
            repo_id,
            "policy_preprocessor.json",
            revision,
        )
    )

    post_json_path = (
        download_model_file(
            repo_id,
            "policy_postprocessor.json",
            revision,
        )
    )

    pre_json = load_json(
        pre_json_path
    )

    post_json = load_json(
        post_json_path
    )

    inspect_processor_json(
        "POLICY PREPROCESSOR JSON",
        pre_json,
    )

    inspect_processor_json(
        "POLICY POSTPROCESSOR JSON",
        post_json,
    )

    # ========================================================
    # Discover state files from JSON itself
    # ========================================================

    _, normalizer_step = (
        find_processor_step(
            pre_json,
            "normalizer",
        )
    )

    _, unnormalizer_step = (
        find_processor_step(
            post_json,
            "unnormalizer",
        )
    )

    if normalizer_step is None:
        raise RuntimeError(
            "Could not find normalizer step "
            "in policy_preprocessor.json"
        )

    if unnormalizer_step is None:
        raise RuntimeError(
            "Could not find unnormalizer step "
            "in policy_postprocessor.json"
        )

    normalizer_filename = (
        normalizer_step.get(
            "state_file"
        )
    )

    unnormalizer_filename = (
        unnormalizer_step.get(
            "state_file"
        )
    )

    if not normalizer_filename:
        raise RuntimeError(
            "Normalizer state_file missing."
        )

    if not unnormalizer_filename:
        raise RuntimeError(
            "Unnormalizer state_file missing."
        )

    normalizer_path = (
        download_model_file(
            repo_id,
            normalizer_filename,
            revision,
        )
    )

    unnormalizer_path = (
        download_model_file(
            repo_id,
            unnormalizer_filename,
            revision,
        )
    )

    # ========================================================
    # Inspect raw safetensors
    # ========================================================

    normalizer_state = (
        inspect_safetensors(
            "RAW NORMALIZER SAFETENSORS",
            normalizer_path,
        )
    )

    unnormalizer_state = (
        inspect_safetensors(
            "RAW UNNORMALIZER SAFETENSORS",
            unnormalizer_path,
        )
    )

    analyze_expected_keys(
        "NORMALIZER KEY ANALYSIS",
        normalizer_state,
    )

    analyze_expected_keys(
        "UNNORMALIZER KEY ANALYSIS",
        unnormalizer_state,
    )

    # ========================================================
    # Load actual policy
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING ACTUAL SmolVLA POLICY")
    print("=" * 70)

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            repo_id,
            revision=revision,
        )
        .to(device)
        .eval()
    )

    print(
        "POLICY_LOAD_OK"
    )

    # ========================================================
    # Build same processor stack used in your inference script
    # ========================================================

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy.config,
            repo_id,
            preprocessor_overrides={
                "device_processor": {
                    "device": device,
                }
            },
        )
    )

    print(
        "PROCESSOR_STACK_LOAD_OK"
    )

    inspect_runtime_pipeline(
        preprocessor,
        postprocessor,
    )

    # ========================================================
    # Save machine-readable report
    # ========================================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    report = {
        "model": repo_id,
        "revision": revision,
        "device": device,

        "preprocessor_json": pre_json,
        "postprocessor_json": post_json,

        "normalizer_safetensors": (
            serialize_state_dict(
                normalizer_state
            )
        ),

        "unnormalizer_safetensors": (
            serialize_state_dict(
                unnormalizer_state
            )
        ),

        "normalizer_feature_names": (
            extract_feature_names_from_keys(
                normalizer_state
            )
        ),

        "unnormalizer_feature_names": (
            extract_feature_names_from_keys(
                unnormalizer_state
            )
        ),
    }

    report_path = (
        OUTPUT_DIR
        / f"{timestamp}_smolvla_stats.json"
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
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Report saved to:"
    )

    print(
        report_path
    )

    print()
    print(
        "No robot or camera was connected."
    )

    print(
        "No physical action was executed."
    )


if __name__ == "__main__":
    main()