"""Run deterministic checks and save evidence for this exact checkout, including failures."""
import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import repository_state, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["fast", "cpu", "gpu"], default="fast")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    folder = args.output or ROOT / "reports" / "verification" / f"{args.profile}-{uuid.uuid4().hex[:12]}"
    if folder.exists():
        raise FileExistsError("Evidence directory exists; use a fresh output path.")
    folder.mkdir(parents=True)
    before = repository_state()
    commands = [("repository", [sys.executable, "scripts/check_repository.py"]),
                ("lint", [sys.executable, "-m", "ruff", "check", "minillm", "scripts", "tests"]),
                ("fast-tests", [sys.executable, "-m", "unittest", "tests.test_data", "tests.test_engineering", "-v"])]
    if args.profile in {"cpu", "gpu"}:
        commands += [("dependencies", [sys.executable, "-m", "pip", "check"]),
                     ("cpu-tests", [sys.executable, "-m", "unittest", "tests.test_lab", "tests.test_lifecycle", "-v"])]
    if args.profile == "gpu":
        commands.append(("gpu-lifecycle", [sys.executable, "scripts/gpu_smoke.py", "--output", str(folder / "gpu.json")]))
    results = []
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true"}
    for name, cmd in commands:
        print(f"Running {name} ({args.profile})", flush=True)
        tick = time.monotonic()
        with (folder / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                code = proc.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.write(str(exc))
                code = 124
        results.append({"name": name, "command": cmd, "exit_code": code,
                        "seconds": time.monotonic() - tick, "log": f"{name}.log"})
        if code:
            print((folder / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-12000:])
            break
    after = repository_state()
    unchanged = before == after
    passed = len(results) == len(commands) and all(r["exit_code"] == 0 for r in results) and unchanged
    write_json(folder / "summary.json", {"schema_version": 1, "profile": args.profile, "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "repository": before, "repository_after": after, "source_unchanged": unchanged,
               "passed": passed, "checks": results,
               "limits": "Engineering verification is not human approval or a model-quality release decision."})
    print(f"{'PASS' if passed else 'FAIL'}: {folder / 'summary.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
