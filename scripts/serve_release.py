"""Serve a checked candidate on localhost; runtime logs stay outside immutable weights."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import read_json
from minillm.release import validate_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    folder, logs = args.bundle.resolve(), args.logs.resolve()
    if logs.is_relative_to(folder):
        raise ValueError("Runtime logs must be outside the immutable bundle")
    manifest = validate_bundle(folder, read_json(args.policy))
    import uvicorn
    from minillm.api import create_app
    cfg = read_json(folder / "config.json")
    uvicorn.run(create_app(folder, cfg, manifest["stage"], log_path=logs / "api_requests.jsonl",
                          release_id=manifest["release_id"]), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
