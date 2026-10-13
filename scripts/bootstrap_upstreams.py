"""Prepare only the two locked upstream repositories; never resets an existing checkout."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(["git", *map(str, args)], check=True)


def main():
    lock = json.loads((ROOT / "upstream-lock.json").read_text())
    for name, entry in lock.items():
        target = ROOT / "vendor" / name
        if not target.exists():
            run("clone", "--no-checkout", "--depth", "1", entry["url"], target)
            run("-C", target, "fetch", "--depth", "1", "origin", entry["commit"])
            run("-C", target, "checkout", "--detach", entry["commit"])
        head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if head != entry["commit"] or dirty:
            raise ValueError(f"{name}: existing checkout differs; preserve it and resolve explicitly")


if __name__ == "__main__":
    main()
