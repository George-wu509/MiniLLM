"""Compare only runs with identical token/evaluation units; exit nonzero on incompatibility."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import read_json, run_path, sha256, write_json
from minillm.data import verify_prepared


def compare_runs(baseline, candidate, stage):
    problems, rows = [], []
    manifests, configs = [], []
    for run in (baseline, candidate):
        manifests.append(verify_prepared(run))
        configs.append(read_json(run / "config.json"))
        row = read_json(run / "evaluation" / f"{stage}.json")
        weight = "adapter_model.safetensors" if stage == "lora" else "model.safetensors"
        if row.get("model_weights_sha256") != sha256(run / stage / weight):
            problems.append(f"{run.name}: missing/stale weight identity in evaluation; re-evaluate explicitly")
        rows.append(row)
    if manifests[0]["tokenizer"]["sha256"] != manifests[1]["tokenizer"]["sha256"]:
        problems.append("Different tokenizers: token-level loss/perplexity units differ")
    for kind in ("pretrain", "sft", "dpo", "domain"):
        if sha256(baseline / "data" / kind / "validation.jsonl") != sha256(candidate / "data" / kind / "validation.jsonl"):
            problems.append(f"Different {kind} validation data")
    for key in ("block_size", "eval_batches"):
        if configs[0][key] != configs[1][key]:
            problems.append(f"Different {key}: raw evaluation windows/protocol differ")
    a, b = (row.get("evaluation_provenance", {}).get("versions") for row in rows)
    if a != b or a is None:
        problems.append("Different/missing evaluation library versions; verify protocol compatibility")
    metrics = ("raw_validation_loss", "raw_perplexity", "sft_completion_loss", "domain_completion_loss",
               "preference_chosen_win_rate", "preference_mean_logp_margin")
    return {"baseline": baseline.name, "candidate": candidate.name, "stage": stage, "compatible": not problems,
            "problems": problems, "metrics": {key: {"baseline": rows[0][key], "candidate": rows[1][key],
                "candidate_minus_baseline": rows[1][key] - rows[0][key]} for key in metrics} if not problems else {},
            "limits": "Comparable proxy metrics do not establish task accuracy or authorize release."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", choices=["base", "sft", "dpo", "lora", "merged"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Comparison evidence exists; use a fresh path")
    result = compare_runs(run_path(args.baseline), run_path(args.candidate), args.stage)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
