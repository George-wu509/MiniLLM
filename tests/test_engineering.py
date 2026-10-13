import copy
import json
import tempfile
import unittest
from pathlib import Path

from minillm.common import ROOT, sha256, write_json
from minillm.data import verify_prepared
from minillm.release import identity, validate_bundle
from scripts.check_repository import validate_config


class EngineeringTests(unittest.TestCase):
    def temporary(self):
        root = ROOT / ".cache" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_context_and_dimensions_validation(self):
        cfg = json.loads((ROOT / "configs" / "ci.json").read_text())
        validate_config(cfg)
        bad = copy.deepcopy(cfg)
        bad["model"]["max_position_embeddings"] = bad["block_size"]
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(cfg)
        bad["model"]["num_key_value_heads"] = 3
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_prepared_data_tampering_and_path_escape(self):
        with self.temporary() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "tokenizer").mkdir()
            data, tok = root / "data" / "train.bin", root / "tokenizer" / "tokenizer.json"
            data.write_bytes(b"original")
            tok.write_text("{}")
            manifest = {"prepared_sha256": {"data/train.bin": sha256(data)}, "tokenizer": {"sha256": sha256(tok)}}
            write_json(root / "data" / "manifest.json", manifest)
            verify_prepared(root)
            data.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_prepared(root)
            manifest["prepared_sha256"] = {"../escape": "x"}
            write_json(root / "data" / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                verify_prepared(root)

    def test_release_hash_binding_quality_and_unexpected_files(self):
        with self.temporary() as temp:
            root = Path(temp)
            files = ["config.json", "data/manifest.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                     "sft/model.safetensors", "sft/config.json", "sft/lineage.json"]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            evaluation = {"stage": "sft", "model_weights_sha256": sha256(root / "sft/model.safetensors"),
                          "tokenizer_sha256": sha256(root / "tokenizer/tokenizer.json"),
                          "dataset_manifest_sha256": sha256(root / "data/manifest.json"),
                          "raw_validation_loss": 4, "raw_perplexity": 54, "sft_completion_loss": 3,
                          "domain_completion_loss": 4, "preference_chosen_win_rate": 0.7, "cache_greedy_equal": True}
            write_json(root / "evaluation.json", evaluation)
            files.append("evaluation.json")
            manifest = {"schema_version": 1, "status": "candidate-not-approved", "stage": "sft",
                        "files": {name: sha256(root / name) for name in files}}
            manifest["release_id"] = identity(manifest)
            write_json(root / "release.json", manifest)
            validate_bundle(root, {"purpose": "test", "metrics": {"raw_validation_loss": {"max": 5}}})
            with self.assertRaisesRegex(ValueError, "threshold"):
                validate_bundle(root, {"purpose": "test", "metrics": {"raw_validation_loss": {"max": 2}}})
            # Manifest identity does not by itself allow arbitrary paths.
            malicious = copy.deepcopy(manifest)
            malicious["files"]["../outside"] = "x"
            malicious["release_id"] = identity(malicious)
            write_json(root / "release.json", malicious)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                validate_bundle(root)
            write_json(root / "release.json", manifest)
            (root / "unexpected.pt").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                validate_bundle(root)
            (root / "unexpected.pt").unlink()
            (root / "sft/model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                validate_bundle(root)
