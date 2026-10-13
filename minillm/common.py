import hashlib
import json
import os
import random
import subprocess
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# All caches are local to this laboratory. No login or remote tracking is required.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed, configure_threads=True):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if configure_threads:
        torch.set_num_threads(min(8, os.cpu_count() or 1))


def device_for(cfg):
    import torch
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run doctor or select device=cpu.")
    return torch.device(device)


def precision_for(device):
    import torch
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def run_path(name):
    # Prevent accidentally putting experiments outside the workspace.
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError("Run name must contain only letters, digits, '-' or '_'.")
    return ROOT / "artifacts" / name


def repository_state(root=ROOT):
    """Identify both committed and uncommitted source, without recording source contents."""
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        branch = git("branch", "--show-current").decode().strip() or "detached"
        dirty = bool(git("status", "--porcelain"))
        names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "source_sha256": None}
    digest = hashlib.sha256()
    for name in sorted(set(filter(None, names))):
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((sha256(path) if path.is_file() else "missing").encode() + b"\0")
    return {"commit": commit, "branch": branch, "dirty": dirty, "source_sha256": digest.hexdigest()}


def provenance():
    import importlib.metadata as md
    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "datasets", "tokenizers", "accelerate"):
        versions[package] = md.version(package)
    commits = {}
    for name in ("minimind", "nanogpt"):
        commits[name] = subprocess.check_output(["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True).strip()
    return {"utc": datetime.now(timezone.utc).isoformat(), "versions": versions, "upstream_commits": commits,
            "repository": repository_state(), "python": sys.version, "platform": platform.platform(),
            "dependency_lock_sha256": sha256(ROOT / "requirements.lock.txt"),
            "lock_files_sha256": {name: sha256(ROOT / name) for name in
                                  ("requirements.lock.txt", "requirements-cpu.lock.txt", "requirements-dev.txt")
                                  if (ROOT / name).is_file()}}


def doctor():
    import psutil
    import torch
    result = provenance()
    result.update({"ram_gb": round(psutil.virtual_memory().total / 2**30, 2), "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda})
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        result.update({"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 2), "bf16": torch.cuda.is_bf16_supported()})
        # Actually execute a CUDA operation, rather than relying only on the driver.
        result["cuda_matmul_ok"] = bool(torch.isfinite(torch.randn(32, 32, device="cuda") @ torch.randn(32, 32, device="cuda")).all())
    write_json(ROOT / "reports" / "environment.json", result)
    print(json.dumps(result, indent=2))
    return result
