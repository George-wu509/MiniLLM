import hashlib
import re
import unicodedata
from pathlib import Path

from .common import ROOT, jsonl, write_json, write_jsonl, sha256

CHAT_TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"


def clean(text):
    if not isinstance(text, str):
        raise ValueError("Dataset fields must be strings.")
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_rows(path, kind):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".jsonl", ".parquet"}) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No supported dataset files under {path}")
    for file in files:
        if file.suffix.lower() == ".txt":
            if kind != "pretrain":
                raise ValueError("TXT is only supported for pretraining. Use JSONL or Parquet for pairs.")
            for text in re.split(r"\n\s*\n", file.read_text(encoding="utf-8-sig")):
                yield {"text": text}
        elif file.suffix.lower() == ".jsonl":
            yield from jsonl(file)
        elif file.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        else:
            raise ValueError(f"Unsupported file: {file}")


def canonical_row(row, kind):
    if kind == "pretrain":
        return {"text": clean(row["text"])}
    if kind in ("sft", "domain"):
        if "messages" in row or "conversations" in row:
            messages = row.get("messages", row.get("conversations"))
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError("SFT messages must end with an assistant response.")
            result = [{"role": m["role"], "content": clean(m["content"])} for m in messages]
            if any(m["role"] not in {"system", "user", "assistant"} for m in result):
                raise ValueError("Only system/user/assistant messages are supported.")
            return {"messages": result}
        prompt = row.get("prompt", row.get("instruction", ""))
        if row.get("input"):
            prompt += "\n" + row["input"]
        answer = row.get("response", row.get("output", row.get("completion", "")))
        return {"messages": [{"role": "user", "content": clean(prompt)}, {"role": "assistant", "content": clean(answer)}]}
    result = {key: clean(row[key]) for key in ("prompt", "chosen", "rejected")}
    if result["chosen"] == result["rejected"]:
        raise ValueError("DPO chosen and rejected must differ.")
    return result


def split_rows(rows, kind, seed):
    import json
    seen, groups = set(), {}
    raw_count = empty_count = 0
    for row in rows:
        raw_count += 1
        row = canonical_row(row, kind)
        values = [m["content"] for m in row["messages"]] if "messages" in row else list(row.values())
        if any(not v for v in values):
            empty_count += 1
            continue
        signature = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        # Keep all answers/preferences for one prompt on one side of the split.
        key = row["text"] if kind == "pretrain" else row["prompt"] if kind == "dpo" else json.dumps(row["messages"][:-1], sort_keys=True, ensure_ascii=False)
        # A single-turn user prompt uses the same key for SFT/domain/DPO.
        if "messages" in row and len(row["messages"]) == 2 and row["messages"][0]["role"] == "user":
            key = row["messages"][0]["content"]
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError(f"{kind}: need at least two unique documents/prompt groups.")
    train, val = [], []
    for key, group in groups.items():
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        (val if int.from_bytes(digest[:8], "big") % 10 == 0 else train).extend(group)
    if not train or not val:
        raise ValueError(f"{kind}: hash split produced an empty partition. Add more unique examples or choose another seed.")
    return train, val, {"raw": raw_count, "empty_removed": empty_count, "duplicates_removed": raw_count - empty_count - len(seen), "train": len(train), "validation": len(val), "groups": len(groups)}


def demo_data(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Demo dataset already exists: {destination}. Use it or choose another destination.")
    raw, sft, dpo, domain = [], [], [], []
    facts = [
        ("Transformer", "Transformer uses attention to connect tokens in a sequence."),
        ("預訓練", "預訓練從隨機權重開始,透過預測下一個 token 學習語言。"),
        ("SFT", "SFT 使用指令與回答資料,讓模型學習遵循指令。"),
        ("LoRA", "LoRA 凍結原始權重,只訓練低秩矩陣,減少可訓練參數。"),
        ("DPO", "DPO 使用偏好配對,比較 chosen 與 rejected 回答。"),
        ("KV cache", "KV cache reuses attention keys and values during autoregressive generation."),
        ("validation", "Validation data is held out from training to estimate generalization."),
        ("optimizer", "AdamW updates parameters using gradients and decoupled weight decay."),
    ]
    for i in range(160):
        topic, answer = facts[i % len(facts)]
        raw.append({"text": f"Laboratory note {i}. {answer} A checkpoint preserves model parameters. Dataset cleaning removes empty documents and exact duplicates. Experiment {i} observes training loss, gradients, and validation loss."})
    for a in range(1, 16):
        for b in range(1, 9):
            prompt = f"Calculate {a} + {b}. Answer with a number."
            answer = str(a + b)
            sft.append({"instruction": prompt, "output": answer})
            dpo.append({"prompt": prompt, "chosen": answer, "rejected": str(a + b + 3)})
            raw.append({"text": f"Arithmetic example: {a} plus {b} equals {a+b}. We add two numbers and write their sum as {a+b}."})
    for i in range(80):
        topic, answer = facts[i % len(facts)]
        domain.append({"instruction": f"Lab question {i}: explain {topic}.", "output": answer})
    # Intentionally include dirty and duplicate records to exercise preparation.
    raw.extend([raw[0], {"text": "   "}, {"text": "\x00 A clean laboratory keeps reproducible records. \r\n"}])
    for name, rows in (("pretrain", raw), ("sft", sft), ("dpo", dpo), ("domain", domain)):
        write_jsonl(destination / f"{name}.jsonl", rows)
    write_json(destination / "source.json", {"origin": "Locally generated educational synthetic data", "license": "CC0-1.0", "purpose": "Pipeline verification; not a useful language model corpus.", "no_downloaded_model_weights": True})


def prepare(run, cfg, paths):
    if (run / "data").exists() or (run / "tokenizer").exists():
        raise FileExistsError("Prepared data/tokenizer already exist. Choose a new --run to preserve provenance.")
    manifest = {"seed": cfg["seed"], "split": "SHA256(seed:normalized document/prompt) modulo 10; validation bucket 0", "deduplication": "exact normalized records; prompt grouping", "sources": {}, "counts": {}}
    splits = {}
    for kind, path in paths.items():
        splits[kind] = split_rows(read_rows(path, kind), kind, cfg["seed"])
        manifest["counts"][kind] = splits[kind][2]
        source = Path(path)
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in {".txt", ".jsonl", ".parquet"}) if source.is_dir() else [source]
        manifest["sources"][kind] = [{"path": str(p.resolve()), "sha256": sha256(p)} for p in files]
    # Train tokenizer strictly on training raw text, never validation or answers.
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator((r["text"] for r in splits["pretrain"][0]), trainers.BpeTrainer(vocab_size=cfg["vocab_size"], min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|pad|>", "<|bos|>", "<|im_end|>", "<|unk|>", "<|im_start|>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<|pad|>", bos_token="<|bos|>", eos_token="<|im_end|>", unk_token="<|unk|>", additional_special_tokens=["<|im_start|>"], model_max_length=cfg["model"]["max_position_embeddings"], chat_template=CHAT_TEMPLATE)
    tokenizer.save_pretrained(run / "tokenizer")
    import numpy as np
    for kind, (train, val, _) in splits.items():
        for split, rows in (("train", train), ("validation", val)):
            write_jsonl(run / "data" / kind / f"{split}.jsonl", rows)
            if kind == "pretrain":
                output = run / "data" / kind / f"{split}.bin"
                count = 0
                with output.open("wb") as handle:
                    for row in rows:
                        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
                        np.asarray(ids, dtype=np.int32).tofile(handle)
                        count += len(ids)
                if count <= cfg["block_size"]:
                    raise ValueError(f"Not enough {split} tokens for block_size={cfg['block_size']}")
                manifest.setdefault("tokens", {})[split] = count
    manifest["tokenizer"] = {"requested_vocab": cfg["vocab_size"], "actual_vocab": len(tokenizer), "training_partition": "pretrain/train", "byte_level": True, "sha256": sha256(run / "tokenizer" / "tokenizer.json")}
    manifest["prepared_sha256"] = {str(p.relative_to(run)): sha256(p) for p in sorted((run / "data").rglob("*")) if p.is_file()}
    write_json(run / "data" / "manifest.json", manifest)
    write_json(run / "config.json", cfg)
    print(manifest["counts"])
    return manifest


def prompt_completion(tokenizer, row, max_length):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Preserve completion tokens and at least one prompt token. No silent all-masked rows.
    if len(prompt_ids) + len(response_ids) > max_length:
        raise ValueError("SFT example exceeds block_size. Increase context or shorten the data; truncation is not silent.")
    return {"input_ids": prompt_ids + response_ids, "completion_mask": [0] * len(prompt_ids) + [1] * len(response_ids)}
