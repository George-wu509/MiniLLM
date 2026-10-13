"""Immutable inference candidates. Hash validation is integrity, not human approval."""
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .common import read_json, sha256, write_json
from .data import verify_prepared

FULL_STAGES = {"base", "sft", "dpo", "merged"}


def identity(manifest):
    payload = {k: v for k, v in manifest.items() if k != "release_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_bundle(folder, policy=None):
    folder = Path(folder).resolve()
    if (folder / "release.json").is_symlink():
        raise ValueError("Release manifest must be a regular local file")
    manifest = read_json(folder / "release.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "candidate-not-approved":
        raise ValueError("Unsupported release manifest/status")
    if manifest["release_id"] != identity(manifest):
        raise ValueError("Release manifest identity mismatch")
    stage = manifest["stage"]
    if stage not in FULL_STAGES:
        raise ValueError("Only full inference models can be packaged; merge LoRA first")
    files = manifest["files"]
    required = {"config.json", "data/manifest.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                f"{stage}/model.safetensors", f"{stage}/config.json", f"{stage}/lineage.json", "evaluation.json"}
    if not required <= files.keys():
        raise ValueError("Release is missing required inference/lineage files")
    for name, expected in files.items():
        path = folder / name
        if "\\" in name or path.is_symlink() or path.resolve() != path.absolute() or not path.resolve().is_relative_to(folder):
            raise ValueError(f"Unsafe release path: {name}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Release file integrity mismatch: {name}")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != set(files) | {"release.json"}:
        raise ValueError("Unexpected files in immutable release candidate")
    evaluation = read_json(folder / "evaluation.json")
    if evaluation["stage"] != stage or evaluation.get("model_weights_sha256") != files[f"{stage}/model.safetensors"]:
        raise ValueError("Evaluation does not identify these model weights")
    if evaluation.get("tokenizer_sha256") != files["tokenizer/tokenizer.json"] or evaluation.get("dataset_manifest_sha256") != files["data/manifest.json"]:
        raise ValueError("Evaluation tokenizer/dataset identity mismatch")
    for key in ("raw_validation_loss", "raw_perplexity", "sft_completion_loss", "domain_completion_loss", "preference_chosen_win_rate"):
        if not math.isfinite(evaluation[key]):
            raise ValueError(f"Nonfinite evaluation metric: {key}")
    if evaluation.get("cache_greedy_equal") is not True:
        raise ValueError("Cache correctness evidence did not pass")
    if policy is not None:
        if not policy.get("purpose") or not policy.get("metrics"):
            raise ValueError("Quality policy needs purpose and explicit metric thresholds")
        for key, bounds in policy["metrics"].items():
            if key not in evaluation or not bounds or set(bounds) - {"min", "max"}:
                raise ValueError(f"Invalid quality policy metric: {key}")
            value = evaluation[key]
            if "min" in bounds and not value >= bounds["min"] or "max" in bounds and not value <= bounds["max"]:
                raise ValueError(f"Quality threshold failed: {key}={value}, bounds={bounds}")
    return manifest


def bundle_candidate(run, stage, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if stage not in FULL_STAGES:
        raise ValueError("Package base/sft/dpo/merged; merge adapters first")
    if destination.exists() or destination.is_relative_to(run):
        raise ValueError("Choose a new candidate directory outside the source run")
    verify_prepared(run)
    lineage = read_json(run / stage / "lineage.json")
    # Merged models retain the LoRA parent's training provenance.
    if stage == "merged":
        lineage = {**lineage, "training_parent": read_json(run / "lora" / "lineage.json")}
        repository = lineage["training_parent"].get("repository")
    else:
        repository = lineage.get("repository")
    if not repository or not repository.get("commit"):
        raise ValueError("Training provenance lacks a MiniLLM commit. Create a traced run; do not guess historical provenance.")
    evaluation = read_json(run / "evaluation" / f"{stage}.json")
    source_files = {"config.json": run / "config.json", "data/manifest.json": run / "data" / "manifest.json",
                    "evaluation.json": run / "evaluation" / f"{stage}.json"}
    for path in (run / "tokenizer").iterdir():
        if path.is_file():
            source_files[f"tokenizer/{path.name}"] = path
    for name in ("model.safetensors", "config.json", "lineage.json", "final_metrics.json", "weight_stats.json"):
        if (run / stage / name).is_file():
            source_files[f"{stage}/{name}"] = run / stage / name
    manifest = {"schema_version": 1, "status": "candidate-not-approved", "stage": stage,
                "created_utc": datetime.now(timezone.utc).isoformat(), "training_repository": repository,
                "evaluation_repository": evaluation.get("evaluation_provenance", {}).get("repository"),
                "files": {name: sha256(path) for name, path in source_files.items()},
                "limits": "Integrity checks and numeric thresholds do not provide human approval or establish useful-model quality."}
    manifest["release_id"] = identity(manifest)
    destination.mkdir(parents=True)
    for name, source in source_files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(destination / "release.json", manifest)
    # Re-read copied bytes; a failed candidate remains visibly invalid, never approved.
    validate_bundle(destination)
    return manifest
