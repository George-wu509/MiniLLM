import math
import json
import tempfile
import unittest
from pathlib import Path

from minillm.common import ROOT, read_json
from minillm.data import verify_prepared
from minillm.evaluate import compare
from minillm.modeling import load_model, tokenizer_at
from minillm.posttrain import train as posttrain
from minillm.pretrain import train as pretrain
from minillm.release import bundle_candidate, validate_bundle
from tests.fixtures import prepared_fixture
from scripts.compare_runs import compare_runs


class LifecycleTests(unittest.TestCase):
    def test_fresh_cpu_lifecycle_and_merged_adapter(self):
        (ROOT / ".cache" / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache" / "tests") as temporary:
            run, cfg = prepared_fixture(Path(temporary))
            pretrain(run, cfg)
            for stage in ("sft", "dpo", "lora"):
                posttrain(run, cfg, stage)
            pretrain(run, cfg, nano=True)
            model = load_model(run / "lora").merge_and_unload()
            model.save_pretrained(run / "merged")
            tokenizer_at(run).save_pretrained(run / "merged")
            report = compare(run, cfg)
            self.assertEqual(len(report["models"]), 6)
            for row in report["models"]:
                self.assertTrue(math.isfinite(row["raw_validation_loss"]))
                self.assertTrue(row["cache_greedy_equal"])
            self.assertTrue(math.isfinite(read_json(run / "nanogpt" / "final_metrics.json")["raw_validation_loss"]))
            self.assertTrue(compare_runs(run, run, "sft")["compatible"])
            generations = [json.loads(line) for line in (run / "evaluation" / "generations.jsonl").read_text().splitlines()]
            lora = {r["prompt"]: r["token_ids"] for r in generations if r["stage"] == "lora"}
            merged = {r["prompt"]: r["token_ids"] for r in generations if r["stage"] == "merged"}
            self.assertEqual(lora, merged)
            candidate = Path(temporary) / "candidate"
            bundle_candidate(run, "sft", candidate)
            manifest = validate_bundle(candidate, read_json(ROOT / "configs" / "release-policy.lab.json"))
            self.assertEqual(manifest["status"], "candidate-not-approved")
            self.assertFalse(list(candidate.rglob("*.pt")))
            # Saved candidate is a usable model with the same logits, not only a hash list.
            import torch
            x = torch.tensor([[30, 40, 50]])
            with torch.no_grad():
                torch.testing.assert_close(load_model(run / "sft")(x).logits,
                                           load_model(candidate / "sft")(x).logits, rtol=0, atol=0)
            from fastapi.testclient import TestClient
            from minillm.api import create_app
            logs = Path(temporary) / "runtime" / "requests.jsonl"
            with TestClient(create_app(candidate, cfg, "sft", logs, manifest["release_id"])) as client:
                self.assertEqual(client.get("/health").json()["release_id"], manifest["release_id"])
                response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2})
                self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(logs.read_text().splitlines()[0])["release_id"], manifest["release_id"])
            validate_bundle(candidate)  # Serving did not write into immutable weights.
            evaluation_path = run / "evaluation" / "sft.json"
            stale = read_json(evaluation_path)
            stale["model_weights_sha256"] = "0" * 64
            from minillm.common import write_json
            write_json(evaluation_path, stale)
            self.assertFalse(compare_runs(run, run, "sft")["compatible"])
            with (run / "data" / "pretrain" / "train.bin").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_prepared(run)
            with self.assertRaisesRegex(ValueError, "integrity"):
                pretrain(run, cfg, resume=run / "base" / "checkpoint-1")
