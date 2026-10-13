"""Explicit, bounded GPU validation in a temporary run; never overwrites lab artifacts."""
import argparse
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from minillm.common import read_json, write_json, provenance
from minillm.data import demo_data, prepare
from minillm.pretrain import train as pretrain
from minillm.posttrain import train as posttrain
from minillm.evaluate import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("GPU evidence already exists.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profile requires CUDA; CPU fallback would invalidate this evidence.")
    cfg = read_json(ROOT / "configs" / "laptop-check.json")
    cfg["device"] = "cuda"
    root = ROOT / ".cache" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        raw, run = Path(temp) / "raw", Path(temp) / "run"
        demo_data(raw)
        prepare(run, cfg, {kind: raw / f"{kind}.jsonl" for kind in ("pretrain", "sft", "dpo", "domain")})
        pretrain(run, cfg)
        for stage in ("sft", "dpo", "lora"):
            posttrain(run, cfg, stage)
        report = compare(run, cfg)
        metrics = {stage: read_json(run / stage / "final_metrics.json") for stage in ("base", "sft", "dpo", "lora")}
        if not all(math.isfinite(row["raw_validation_loss"]) and row["cache_greedy_equal"] for row in report["models"]):
            raise RuntimeError("GPU loss/cache validation failed.")
        write_json(args.output, {"configuration": cfg, "gpu": torch.cuda.get_device_name(0),
                   "provenance": provenance(), "comparison": report, "metrics": metrics,
                   "limits": "Two-step synthetic hardware validation; not useful-model quality evidence."})
    print(f"GPU evidence: {args.output}")


if __name__ == "__main__":
    main()
