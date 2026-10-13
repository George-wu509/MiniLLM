import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

from .common import jsonl, read_json, write_json, sha256, seed_all, device_for, precision_for, provenance
from .data import prompt_completion, verify_prepared
from .modeling import tokenizer_at, load_model, weight_stats


class MetricsCallback(TrainerCallback):
    def __init__(self, folder):
        self.path = folder / "metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, **(logs or {})}, default=str) + "\n")


class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        length = max(len(f["input_ids"]) for f in features)
        ids, masks, labels = [], [], []
        for f in features:
            x, cm = f["input_ids"], f["completion_mask"]
            pad = length - len(x)
            ids.append(x + [self.pad_id]*pad)
            masks.append([1]*len(x) + [0]*pad)
            labels.append([t if m else -100 for t, m in zip(x, cm)] + [-100]*pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}


def completion_ce(outputs, labels, num_items_in_batch=None):
    # Sum across supervised tokens, normalize by the accumulated batch token count.
    # This matches a larger batch even when response lengths differ across microbatches.
    logits = outputs.logits[:, :-1].float().contiguous()
    targets = labels[:, 1:].contiguous()
    denominator = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
                           reduction="sum") / denominator


def train(run, cfg, stage, parent=None, resume=None):
    verify_prepared(run)
    parent = parent or ("sft" if stage in {"dpo", "lora"} else "base")
    if parent not in {"base", "sft", "dpo"} or parent == stage:
        raise ValueError("Parent must be a different full-model stage: base, sft or dpo.")
    target = run / stage
    if (target / "model.safetensors").exists() or (target / "adapter_model.safetensors").exists():
        raise FileExistsError(f"Completed {stage} exists. Choose another run.")
    if target.exists() and (target / "metrics.jsonl").exists() and not resume:
        raise FileExistsError("Partial run exists. Supply --resume or choose a new run.")
    target.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    device = device_for(cfg)
    dtype = precision_for(device)
    tokenizer = tokenizer_at(run)
    model = load_model(run / parent)
    model.config.use_cache = False
    # Prevent Trainer from treating the upstream **kwargs mean CE as a sum loss.
    model.accepts_loss_kwargs = False
    common = dict(output_dir=str(target), max_steps=cfg[f"{stage}_steps"], learning_rate=cfg[f"{stage}_lr"],
                  per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
                  gradient_accumulation_steps=cfg["gradient_accumulation_steps"], max_grad_norm=cfg["max_grad_norm"],
                  weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type="cosine",
                  logging_steps=1, save_steps=cfg["save_steps"], eval_steps=cfg["eval_steps"],
                  eval_strategy="steps", save_strategy="steps", save_total_limit=None,
                  bf16=dtype == torch.bfloat16, fp16=False, use_cpu=device.type == "cpu", tf32=device.type == "cuda",
                  dataloader_num_workers=0, dataloader_pin_memory=device.type == "cuda", report_to=["tensorboard"],
                  logging_dir=str(run / "tensorboard" / stage), seed=cfg["seed"], data_seed=cfg["seed"],
                  gradient_checkpointing=False, remove_unused_columns=False, optim="adamw_torch", disable_tqdm=True)
    callback = MetricsCallback(target)
    peft_cfg = None
    if stage == "dpo":
        def format_pairs(split):
            rows = []
            for row in jsonl(run / "data" / "dpo" / f"{split}.jsonl"):
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
                # DPOTrainer appends EOS itself in this pinned version.
                for response in (row["chosen"], row["rejected"]):
                    if len(tokenizer.encode(prompt, add_special_tokens=False)) + len(tokenizer.encode(response, add_special_tokens=False)) + 1 > cfg["block_size"]:
                        raise ValueError("DPO pair exceeds block_size. Increase context; no silent truncation.")
                rows.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]})
            return Dataset.from_list(rows)
        reference = load_model(run / parent)
        reference.eval()
        reference.requires_grad_(False)
        args = DPOConfig(**common, beta=cfg["dpo_beta"], max_length=cfg["block_size"], max_prompt_length=cfg["block_size"]-1,
                         loss_type="sigmoid", precompute_ref_log_probs=False)
        trainer = DPOTrainer(model=model, ref_model=reference, args=args, processing_class=tokenizer,
                             train_dataset=format_pairs("train"), eval_dataset=format_pairs("validation"), callbacks=[callback])
    else:
        kind = "domain" if stage == "lora" else "sft"
        def processed(split):
            return Dataset.from_list([prompt_completion(tokenizer, row, cfg["block_size"]) for row in jsonl(run / "data" / kind / f"{split}.jsonl")])
        if stage == "lora":
            from peft import LoraConfig
            peft_cfg = LoraConfig(task_type="CAUSAL_LM", r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none")
        args = SFTConfig(**common, max_length=cfg["block_size"], packing=False, completion_only_loss=True,
                         dataset_kwargs={"skip_prepare_dataset": True})
        trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=processed("train"),
                             eval_dataset=processed("validation"), data_collator=CompletionCollator(tokenizer.pad_token_id),
                             peft_config=peft_cfg, callbacks=[callback], compute_loss_func=completion_ce)
    # PEFT wrapper also exposes **kwargs; make accumulation use mean-loss scaling.
    trainer.model_accepts_loss_kwargs = False
    parameters = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    lineage = {"parent": parent, "stage": stage, "parameters": parameters, "trainable_parameters": trainable,
               "trainable_fraction": trainable / parameters, "loss_scope": "preference pairs" if stage == "dpo" else "completion only",
               "configuration": args.to_dict(), "run_configuration": cfg,
               "dataset_manifest_sha256": sha256(run / "data" / "manifest.json"),
               "parent_weights_sha256": sha256(run / parent / "model.safetensors"), **provenance()}
    if resume:
        checkpoint = Path(resume).resolve()
        previous_path = checkpoint.parent / "lineage.json"
        if not previous_path.exists():
            raise ValueError("Resume checkpoint has no stage lineage metadata.")
        previous = read_json(previous_path)
        for key in ("parent", "stage", "run_configuration", "dataset_manifest_sha256", "parent_weights_sha256"):
            if key not in previous or previous[key] != lineage[key]:
                raise ValueError(f"Resume checkpoint lineage mismatch: {key}. Use its original run/data/parent/configuration.")
        if stage in {"sft", "lora"}:
            history = read_json(checkpoint / "trainer_state.json")["log_history"]
            trainer._total_train_tokens = max((row.get("num_tokens", 0) for row in history), default=0)
    write_json(target / "lineage.json", lineage)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(Path(resume).resolve()) if resume else None)
    metrics = {**result.metrics, **trainer.evaluate()}
    logged_losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    metrics["mean_logged_loss_over_completed_steps"] = sum(logged_losses)/len(logged_losses) if logged_losses else None
    if resume:
        metrics["resumed_from"] = str(Path(resume).resolve())
        metrics["train_loss_note"] = "HF train_loss on resume covers this invocation divided by global steps; use mean_logged_loss_over_completed_steps or individual logged losses for the full trajectory."
    metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0
    trainer.save_model(str(target))
    tokenizer.save_pretrained(target)
    trainer.save_state()
    write_json(target / "final_metrics.json", metrics)
    write_json(target / "weight_stats.json", weight_stats(trainer.model))
    # All stages retain the exact parent/data/configuration identity for resuming.
    return target
