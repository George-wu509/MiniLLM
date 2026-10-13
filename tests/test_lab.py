import copy
import faulthandler
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from minillm.common import ROOT, read_json, seed_all
from minillm.data import clean, split_rows, prompt_completion, read_rows
from minillm.modeling import fresh_model, load_model, tokenizer_at
from minillm.posttrain import CompletionCollator, completion_ce
from tests.fixtures import model_fixture


class LaboratoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faulthandler.dump_traceback_later(120)
        test_root = ROOT / ".cache" / "tests"
        test_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=test_root)
        cls.temp = Path(cls.temporary.name)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.addClassCleanup(faulthandler.cancel_dump_traceback_later)
        cls.lab_run, cls.cfg = model_fixture(cls.temp / "fixture")
        cls.tokenizer = tokenizer_at(cls.lab_run)
        seed_all(42)
        cls.model = fresh_model(cls.cfg, cls.tokenizer).eval()

    @classmethod
    def tearDownClass(cls):
        faulthandler.cancel_dump_traceback_later()
        cls.temporary.cleanup()

    def test_clean_dedup_and_group_split(self):
        self.assertEqual(clean("  Ａ\x00\r\n  B  "), "A\nB")
        rows = [{"instruction": f"Question {i}", "output": "answer"} for i in range(100)]
        rows += [rows[0], {"instruction": "Question 0", "output": "another answer"}]
        train, val, counts = split_rows(rows, "sft", 42)
        keys = lambda items: {r["messages"][0]["content"] for r in items}
        self.assertFalse(keys(train) & keys(val))
        self.assertEqual(counts["duplicates_removed"], 1)
        self.assertEqual((train, val), split_rows(rows, "sft", 42)[:2])
        pairs = [{"prompt": f"Question {i}", "chosen": "answer", "rejected": "wrong"} for i in range(100)]
        pt, pv, _ = split_rows(pairs, "dpo", 42)
        self.assertEqual(keys(train), {r["prompt"] for r in pt})
        self.assertEqual(keys(val), {r["prompt"] for r in pv})

    def test_byte_tokenizer_and_chat_template(self):
        text = "未見過的漢字🧪 café"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertNotIn(self.tokenizer.unk_token_id, ids)
        self.assertEqual(self.tokenizer.decode(ids), text)
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
        self.assertEqual(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")

    def test_jsonl_txt_parquet_input(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        txt = self.temp / "raw.txt"
        txt.write_text("first document\n\nsecond document", encoding="utf-8")
        self.assertEqual(len(list(read_rows(txt, "pretrain"))), 2)
        parquet = self.temp / "raw.parquet"
        pq.write_table(pa.Table.from_pylist([{"text": "one"}, {"text": "two"}]), parquet)
        self.assertEqual(list(read_rows(parquet, "pretrain")), [{"text": "one"}, {"text": "two"}])

    def test_completion_and_padding_masks(self):
        features = [prompt_completion(self.tokenizer, {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": answer}]}, 128) for answer in ("5", "A longer answer")]
        batch = CompletionCollator(self.tokenizer.pad_token_id)(features)
        for i, f in enumerate(features):
            supervised = batch["labels"][i] != -100
            self.assertEqual(supervised.sum().item(), sum(f["completion_mask"]))
            self.assertTrue((batch["labels"][i][batch["attention_mask"][i] == 0] == -100).all())
        labels = batch["labels"]
        logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        actual = completion_ce(SimpleNamespace(logits=logits), labels)
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected)

    def test_accumulation_matches_full_batch_with_different_response_lengths(self):
        torch.manual_seed(9)
        logits = torch.randn(2, 6, 17, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 3, 4, -100], [-100, -100, 2, 3, 4, 5]])
        full = completion_ce(SimpleNamespace(logits=logits), labels)
        full_grad = torch.autograd.grad(full, logits, retain_graph=True)[0]
        denominator = (labels[:, 1:] != -100).sum()
        accumulated = sum(completion_ce(SimpleNamespace(logits=logits[i:i+1]), labels[i:i+1], denominator) for i in range(2))
        accumulation_grad = torch.autograd.grad(accumulated, logits)[0]
        torch.testing.assert_close(full_grad, accumulation_grad)

    def test_causal_attention_and_cached_logits(self):
        x = torch.tensor([[30, 40, 50, 60, 70]])
        changed = x.clone()
        changed[0, -1] = 71
        with torch.no_grad():
            full = self.model(x).logits
            altered = self.model(changed).logits
            prefix = self.model(x[:, :-1], use_cache=True)
            cached = self.model(x[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits
        torch.testing.assert_close(full[:, :-1], altered[:, :-1])
        torch.testing.assert_close(full[:, -1:], cached, rtol=1e-5, atol=1e-6)

    def test_save_reload_preserves_weights_and_logits(self):
        path = self.temp / "saved"
        self.model.save_pretrained(path)
        restored = load_model(path).eval()
        x = torch.tensor([[30, 40, 50]])
        with torch.no_grad():
            torch.testing.assert_close(self.model(x).logits, restored(x).logits, rtol=0, atol=0)

    def test_lora_updates_only_adapter(self):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(copy.deepcopy(self.model), LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        x = torch.tensor([[30, 40, 50, 60]])
        model(x, labels=x).loss.backward()
        optim.step()
        changed = []
        for name, p in model.named_parameters():
            if not torch.equal(p, before[name]):
                changed.append(name)
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, before[name]))
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in name for name in changed))

    def test_checkpoint_optimizer_rng_resume_matches_uninterrupted(self):
        from minillm.pretrain import train
        cfg = copy.deepcopy(self.cfg)
        cfg.update(device="cpu", pretrain_steps=4, save_steps=2, eval_steps=2, eval_batches=2, batch_size=2, gradient_accumulation_steps=2)
        uninterrupted, resumed = self.temp / "uninterrupted", self.temp / "resumed"
        for run in (uninterrupted, resumed):
            for folder in ("data", "tokenizer"):
                shutil.copytree(self.lab_run / folder, run / folder)
        train(uninterrupted, cfg)
        train(resumed, cfg, resume=uninterrupted / "base" / "checkpoint-2")
        a, b = load_model(uninterrupted / "base"), load_model(resumed / "base")
        for name, p in a.state_dict().items():
            torch.testing.assert_close(p, b.state_dict()[name], rtol=0, atol=0)

    def test_trl_sft_dpo_lora_resume_matches_uninterrupted(self):
        from minillm.posttrain import train
        cfg = copy.deepcopy(self.cfg)
        cfg.update(device="cpu", sft_steps=4, dpo_steps=4, lora_steps=4, save_steps=2, eval_steps=2,
                   batch_size=2, gradient_accumulation_steps=2)
        for stage in ("sft", "dpo", "lora"):
            with self.subTest(stage=stage):
                full, resumed = self.temp / f"{stage}-full", self.temp / f"{stage}-resumed"
                parent = "base" if stage == "sft" else "sft"
                for run in (full, resumed):
                    for folder in ("data", "tokenizer", parent):
                        shutil.copytree(self.lab_run / folder, run / folder)
                train(full, cfg, stage)
                train(resumed, cfg, stage, resume=full / stage / "checkpoint-2")
                a, b = load_model(full / stage), load_model(resumed / stage)
                for name, p in a.state_dict().items():
                    torch.testing.assert_close(p, b.state_dict()[name], rtol=0, atol=0)
                if stage in {"sft", "lora"}:
                    self.assertEqual(read_json(full / stage / "final_metrics.json")["eval_num_tokens"],
                                     read_json(resumed / stage / "final_metrics.json")["eval_num_tokens"])

    def test_api_health_generation_metrics_and_invalid_requests(self):
        from fastapi.testclient import TestClient
        from minillm.api import create_app
        cfg = {**self.cfg, "device": "cpu"}
        api_run = self.temp / "api"
        for folder in ("tokenizer", "dpo"):
            shutil.copytree(self.lab_run / folder, api_run / folder)
        client = TestClient(create_app(api_run, cfg, "dpo"))
        self.assertEqual(client.get("/health").json()["status"], "ok")
        result = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["usage"]["total_tokens"], result.json()["usage"]["prompt_tokens"]+result.json()["usage"]["completion_tokens"])
        self.assertEqual(client.get("/metrics").json()["requests"], 1)
        self.assertEqual(client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "stream": True}).status_code, 400)
        self.assertEqual(client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "top_p": 0}).status_code, 422)
        from minillm.inference import Engine
        with self.assertRaises(ValueError):
            Engine(api_run, cfg, "dpo").generate("x"*2000)


if __name__ == "__main__":
    unittest.main()
