"""Dependency-free syntax/configuration checks used by both local verification and CI."""
import ast
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_config(cfg):
    positive = ("vocab_size", "block_size", "batch_size", "gradient_accumulation_steps", "pretrain_steps",
                "sft_steps", "dpo_steps", "lora_steps", "save_steps", "eval_steps", "eval_batches", "lora_r")
    for key in positive:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    for key in ("pretrain_lr", "sft_lr", "dpo_lr", "lora_lr", "max_grad_norm", "lora_alpha", "dpo_beta"):
        if not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < float("inf"):
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["warmup_ratio"] <= 1 or not 0 <= cfg["weight_decay"] < float("inf"):
        raise ValueError("Invalid warmup_ratio or weight_decay")
    model = cfg["model"]
    if any(type(value) is not int or value <= 0 for value in model.values()):
        raise ValueError("Model dimensions must be positive integers")
    h, heads, kv = (model[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads"))
    if h % heads or heads % kv or (h // heads) % 2:
        raise ValueError("Invalid attention/GQA/RoPE dimensions")
    if cfg["block_size"] + 1 > model["max_position_embeddings"]:
        raise ValueError("Context must fit block_size + one next-token target")


def check(root=ROOT):
    for folder in ("minillm", "scripts", "tests"):
        for path in (root / folder).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in (root / "configs").glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if "model" in cfg:
            validate_config(cfg)
    for path in (root / ".claude").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"))
    tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    import yaml
    for path in (root / ".github").rglob("*.yml"):
        yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    cuda_pins = {line for line in (root / "requirements.lock.txt").read_text().splitlines() if line and not line.startswith("torch==")}
    cpu_pins = {line for line in (root / "requirements-cpu.lock.txt").read_text().splitlines() if line and not line.startswith("#")}
    if cuda_pins != cpu_pins:
        raise ValueError("CPU/CUDA non-torch pins drifted; update both lock files together")
    lock = json.loads((root / "upstream-lock.json").read_text())
    for name, expected in lock.items():
        repo = root / "vendor" / name
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
        if head != expected["commit"] or dirty:
            raise ValueError(f"Pinned upstream differs or has edits: {name}")
    print("Python syntax, configuration and pinned upstream checks passed.")


if __name__ == "__main__":
    check()
