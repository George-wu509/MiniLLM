"""Fast tests: no torch, tokenizer, existing artifacts or GPU needed."""
import unittest

from minillm.data import canonical_row, clean, split_rows


class DataTests(unittest.TestCase):
    def test_instruction_input_conversion(self):
        row = canonical_row({"instruction": "Translate", "input": "你好", "output": "Hello"}, "sft")
        self.assertEqual(row, {"messages": [{"role": "user", "content": "Translate\n你好"},
                                            {"role": "assistant", "content": "Hello"}]})

    def test_dedup_and_cross_stage_grouping(self):
        rows = [{"instruction": f"Question {i}", "output": "answer"} for i in range(100)]
        rows += [rows[0], {"instruction": "Question 0", "output": "another answer"}]
        train, val, counts = split_rows(rows, "sft", 42)
        train_keys = {r["messages"][0]["content"] for r in train}
        val_keys = {r["messages"][0]["content"] for r in val}
        self.assertFalse(train_keys & val_keys)
        self.assertEqual(counts["duplicates_removed"], 1)
        pairs = [{"prompt": f"Question {i}", "chosen": "answer", "rejected": "wrong"} for i in range(100)]
        pt, pv, _ = split_rows(pairs, "dpo", 42)
        self.assertEqual(train_keys, {r["prompt"] for r in pt})
        self.assertEqual(val_keys, {r["prompt"] for r in pv})

    def test_hash_assignment_survives_reordering(self):
        rows = [{"text": f"document {i}"} for i in range(100)]
        a = split_rows(rows, "pretrain", 42)
        b = split_rows(list(reversed(rows)), "pretrain", 42)
        for index in (0, 1):
            self.assertEqual({r["text"] for r in a[index]}, {r["text"] for r in b[index]})

    def test_cleanup_and_invalid_preference(self):
        self.assertEqual(clean(" Ａ\x00\r\n B "), "A\nB")
        with self.assertRaises(ValueError):
            canonical_row({"prompt": "x", "chosen": "same", "rejected": "same"}, "dpo")
