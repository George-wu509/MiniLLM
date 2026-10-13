import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import read_json, run_path
from minillm.release import bundle_candidate, validate_bundle


def main():
    parser = argparse.ArgumentParser(description="Build/check an inference candidate; no upload, approval or deployment")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--run", required=True)
    bundle.add_argument("--stage", choices=["base", "sft", "dpo", "merged"], required=True)
    bundle.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        result = bundle_candidate(run_path(args.run), args.stage, args.destination)
    else:
        result = validate_bundle(args.bundle, read_json(args.policy))
    print(json.dumps({"release_id": result["release_id"], "status": result["status"], "stage": result["stage"]}, indent=2))


if __name__ == "__main__":
    main()
