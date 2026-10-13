import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .common import seed_all, device_for, precision_for, write_json, provenance, sha256
from .data import verify_prepared
from .modeling import fresh_model, tokenizer_at, load_model, weight_stats


def token_array(run, split):
    return np.memmap(run / "data" / "pretrain" / f"{split}.bin", dtype=np.int32, mode="r")


def get_batch(tokens, batch_size, length, device):
    if len(tokens) <= length:
        raise ValueError("Dataset has too few tokens for this context length.")
    starts = torch.randint(len(tokens) - length, (batch_size,)).tolist()
    sequences = torch.tensor(np.stack([tokens[i:i+length+1].astype(np.int64) for i in starts]), device=device)
    # nanoGPT uses Tensor.view(), so sliced targets must be contiguous.
    return sequences[:, :-1].contiguous(), sequences[:, 1:].contiguous()


def autocast(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()


@torch.no_grad()
def raw_loss(model, run, cfg, nano=False):
    model.eval()
    device = next(model.parameters()).device
    data = token_array(run, "validation")
    length = cfg["block_size"]
    losses = []
    # Fixed, deterministic validation windows shared by every stage.
    for start in range(0, min(len(data)-length, cfg["eval_batches"]*length), length):
        tokens = torch.tensor(np.array(data[start:start+length+1], dtype=np.int64), device=device).unsqueeze(0)
        if nano:
            _, loss = model(tokens[:, :-1], tokens[:, 1:])
        else:
            # MiniMind internally shifts labels; the final input token is a target.
            out = model(tokens, labels=tokens, use_cache=False)
            loss = out.loss
        losses.append(loss.item())
    if not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError("Validation loss is missing or nonfinite.")
    return sum(losses) / len(losses)


def initialize(run, cfg):
    verify_prepared(run)
    target = run / "initialized"
    if (target / "model.safetensors").exists():
        return target
    seed_all(cfg["seed"])
    tokenizer = tokenizer_at(run)
    model = fresh_model(cfg, tokenizer)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    write_json(target / "weight_stats.json", weight_stats(model))
    write_json(target / "lineage.json", {"parent": None, "initialization": "normal(0, 0.02) for Linear/Embedding; RMSNorm=1; tied output embedding", "parameters": sum(p.numel() for p in model.parameters()), **provenance()})
    print(f"Saved random initialization: {target}")
    return target


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def train(run, cfg, resume=None, nano=False):
    verify_prepared(run)
    stage = "nanogpt" if nano else "base"
    target = run / stage
    if (target / "model.safetensors").exists() or (target / "final.pt").exists():
        raise FileExistsError(f"Completed {stage} already exists. Choose another run.")
    seed_all(cfg["seed"])
    device = device_for(cfg)
    dtype = precision_for(device)
    tokenizer = tokenizer_at(run)
    if nano:
        import importlib.util
        import sys
        from dataclasses import asdict
        from .common import ROOT
        spec = importlib.util.spec_from_file_location("lab_nanogpt", ROOT / "vendor" / "nanogpt" / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        nano_cfg = module.GPTConfig(block_size=cfg["block_size"], vocab_size=len(tokenizer), n_layer=cfg["model"]["num_hidden_layers"], n_head=cfg["model"]["num_attention_heads"], n_embd=cfg["model"]["hidden_size"], dropout=0.0, bias=False)
        model = module.GPT(nano_cfg).to(device)
        write_json(target / "model_config.json", asdict(nano_cfg))
        if not resume and not (target / "initialized.pt").exists():
            torch.save({"config": asdict(nano_cfg), "model": model.state_dict()}, target / "initialized.pt")
    else:
        initial = initialize(run, cfg)
        model = load_model(Path(resume) if resume else initial, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["pretrain_lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    total = cfg["pretrain_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total, int(total*cfg["warmup_ratio"])))
    start_step = 0
    if resume:
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        if state["config"] != cfg:
            raise ValueError("Resume requires exactly the saved run configuration.")
        if state.get("dataset_manifest_sha256") != sha256(run / "data" / "manifest.json"):
            raise ValueError("Resume dataset/tokenizer provenance differs from the checkpoint.")
        if nano:
            model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif target.exists() and (target / "metrics.jsonl").exists() and (target / "metrics.jsonl").stat().st_size:
        raise FileExistsError("Partial run exists. Use --resume with a checkpoint or choose a new run.")
    target.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    writer = SummaryWriter(str(run / "tensorboard" / stage))
    log = (target / "metrics.jsonl").open("a", encoding="utf-8")
    import json
    validation = raw_loss(model, run, cfg, nano)
    write_json(target / "initial_eval.json", {"raw_validation_loss": validation, "perplexity": math.exp(min(validation, 50))})
    train_tokens = token_array(run, "train")
    started = time.perf_counter()
    last_validation = validation

    def checkpoint(step):
        folder = target / f"checkpoint-{step}"
        folder.mkdir(parents=True, exist_ok=True)
        if not nano:
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
        torch.save({"step": step, "config": cfg, "dataset_manifest_sha256": sha256(run / "data" / "manifest.json"), "model": model.state_dict() if nano else None, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, folder / "training_state.pt")
        write_json(folder / "metrics.json", {"step": step, "raw_validation_loss": last_validation})

    try:
        for step in range(start_step, total):
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(cfg["gradient_accumulation_steps"]):
                x, y = get_batch(train_tokens, cfg["batch_size"], cfg["block_size"], device)
                with autocast(device, dtype):
                    if nano:
                        _, loss = model(x, y)
                    else:
                        # Same L next-token targets as nanoGPT.
                        seq = torch.cat([x, y[:, -1:]], dim=1)
                        loss = model(seq, labels=seq, use_cache=False).loss
                    scaled_loss = loss / cfg["gradient_accumulation_steps"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step+1}")
                loss_value += loss.item() / cfg["gradient_accumulation_steps"]
                scaled_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(grad):
                raise RuntimeError(f"Nonfinite gradient at step {step+1}")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            row = {"step": step+1, "train_loss": loss_value, "grad_norm_before_clip": grad.item(), "learning_rate": lr, "tokens_per_second": cfg["batch_size"]*cfg["block_size"]*cfg["gradient_accumulation_steps"]/max(elapsed, 1e-9)}
            if (step+1) % cfg["eval_steps"] == 0 or step+1 == total:
                last_validation = raw_loss(model, run, cfg, nano)
                row["raw_validation_loss"] = last_validation
                print(f"{stage} step {step+1}/{total} loss={loss_value:.4f} val={last_validation:.4f} grad={grad.item():.3f}", flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            for key, value in row.items():
                if key != "step":
                    writer.add_scalar(key, value, step+1)
            if (step+1) % cfg["save_steps"] == 0 or step+1 == total:
                checkpoint(step+1)
        if nano:
            torch.save({"config": asdict(nano_cfg), "model": model.state_dict()}, target / "final.pt")
            tokenizer.save_pretrained(target)
            model.eval()
            prefix = "Arithmetic example:"
            ids = torch.tensor([tokenizer.encode(prefix, add_special_tokens=False)], device=device)
            with torch.no_grad():
                generated = model.generate(ids, max_new_tokens=32, temperature=0.8, top_k=min(20, len(tokenizer)))
            write_json(target / "generation.json", {"prompt": prefix, "text": tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True), "temperature": 0.8, "top_k": 20, "use_cache": False})
        else:
            model.save_pretrained(target)
            tokenizer.save_pretrained(target)
            write_json(target / "weight_stats.json", weight_stats(model))
        write_json(target / "lineage.json", {"parent": None if nano else "initialized", "objective": "next-token cross entropy", "optimizer": "AdamW", "steps": total, "parameters": sum(p.numel() for p in model.parameters()), "precision": str(dtype), **provenance()})
        write_json(target / "final_metrics.json", {"raw_validation_loss": last_validation, "perplexity": math.exp(min(last_validation, 50)), "seconds": time.perf_counter()-started, "peak_vram_mb": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0})
    finally:
        log.close()
        writer.close()
    return target
