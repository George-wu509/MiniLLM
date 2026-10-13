import gc
import math

import torch
import torch.nn.functional as F

from .common import jsonl, read_json, write_json, write_jsonl, sha256, provenance
from .data import prompt_completion
from .inference import Engine, cache_benchmark
from .pretrain import raw_loss


@torch.no_grad()
def completion_loss(model, tokenizer, rows, max_length):
    total_loss = count = 0
    device = next(model.parameters()).device
    for row in rows:
        item = prompt_completion(tokenizer, row, max_length)
        x = torch.tensor([item["input_ids"]], device=device)
        mask = torch.tensor([item["completion_mask"]], device=device)[:, 1:]
        logits = model(x, use_cache=False).logits[:, :-1].float()
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
        total_loss += (losses * mask.reshape(-1)).sum().item()
        count += mask.sum().item()
    if not count:
        raise ValueError("No completion tokens for evaluation.")
    return total_loss / count


@torch.no_grad()
def response_logp(model, tokenizer, prompt, response):
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    p = tokenizer.encode(prefix, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(p)+len(r) > model.config.max_position_embeddings:
        raise ValueError("Preference example exceeds context.")
    x = torch.tensor([p+r], device=next(model.parameters()).device)
    logits = model(x, use_cache=False).logits[:, len(p)-1:-1].float()
    targets = torch.tensor([r], device=x.device)
    return -F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum").item()


def compare(run, cfg, stages=None):
    stages = stages or [s for s in ("initialized", "base", "sft", "dpo", "lora", "merged") if (run / s / "model.safetensors").exists() or (run / s / "adapter_model.safetensors").exists()]
    if not stages:
        raise ValueError("No completed models to compare.")
    sft_rows = list(jsonl(run / "data" / "sft" / "validation.jsonl"))
    domain_rows = list(jsonl(run / "data" / "domain" / "validation.jsonl"))
    pairs = list(jsonl(run / "data" / "dpo" / "validation.jsonl"))
    prompts = ["Calculate 2 + 3. Answer with a number.", "Explain LoRA.", "說明什麼是預訓練。"]
    summaries, samples = [], []
    for stage in stages:
        engine = Engine(run, cfg, stage)
        model, tokenizer = engine.model, engine.tokenizer
        raw = raw_loss(model, run, cfg)
        margins = [response_logp(model, tokenizer, row["prompt"], row["chosen"])-response_logp(model, tokenizer, row["prompt"], row["rejected"]) for row in pairs]
        summary = {"stage": stage, "raw_validation_loss": raw, "raw_perplexity": math.exp(min(raw, 50)),
                   "model_weights_sha256": sha256(run / stage / ("adapter_model.safetensors" if stage == "lora" else "model.safetensors")),
                   "tokenizer_sha256": sha256(run / "tokenizer" / "tokenizer.json"),
                   "dataset_manifest_sha256": sha256(run / "data" / "manifest.json"),
                   "evaluation_provenance": provenance(),
                   "sft_completion_loss": completion_loss(model, tokenizer, sft_rows, cfg["block_size"]),
                   "domain_completion_loss": completion_loss(model, tokenizer, domain_rows, cfg["block_size"]),
                   "preference_chosen_win_rate": sum(m > 0 for m in margins)/len(margins),
                   "preference_mean_logp_margin": sum(margins)/len(margins),
                   "preference_logp_margins": margins, "parameters": sum(p.numel() for p in model.parameters())}
        for prompt in prompts:
            samples.append(engine.generate(prompt, greedy=True, max_new_tokens=32))
        bench = cache_benchmark(engine)
        summary["cache_greedy_equal"] = bench["greedy_token_ids_equal"]
        summary["cache_speedup"] = bench["speedup"]
        write_json(run / "evaluation" / f"{stage}.json", summary)
        write_json(run / "evaluation" / f"{stage}_cache.json", bench)
        summaries.append(summary)
        print(f"{stage}: raw loss {raw:.4f}; SFT loss {summary['sft_completion_loss']:.4f}; preference win {summary['preference_chosen_win_rate']:.1%}", flush=True)
        del engine, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Updating one stage must preserve the other stages' evaluation results.
    evaluated_stages = {row["stage"] for row in summaries}
    for stage in ("initialized", "base", "sft", "dpo", "lora", "merged"):
        saved = run / "evaluation" / f"{stage}.json"
        if stage not in evaluated_stages and saved.exists():
            summaries.append(read_json(saved))
    order = {name: i for i, name in enumerate(("initialized", "base", "sft", "dpo", "lora", "merged"))}
    summaries.sort(key=lambda row: order[row["stage"]])
    existing_samples = run / "evaluation" / "generations.jsonl"
    if existing_samples.exists():
        samples = [row for row in jsonl(existing_samples) if row["stage"] not in evaluated_stages] + samples
        samples.sort(key=lambda row: order[row["stage"]])
    reference = next((s for s in summaries if s["stage"] == "sft"), None)
    if reference:
        for summary in summaries:
            advantages = [cfg["dpo_beta"]*(m-r) for m, r in zip(summary["preference_logp_margins"], reference["preference_logp_margins"])]
            summary["dpo_reward_margin_vs_sft"] = sum(advantages)/len(advantages)
    report = {"dataset_manifest_sha256": sha256(run / "data" / "manifest.json"), "configuration": cfg,
              "evaluation_precision": "FP32 with TF32 disabled",
              "protocol": "identical raw validation windows, all held-out SFT/domain/pairs; greedy chat prompts; no held-out data used by tokenizer",
              "limits": "Synthetic smoke data tests mechanics. Losses across different objectives are not interchangeable. Raw preference log-prob sums are length-sensitive. Generation quality is not established by this report.", "models": summaries}
    if (run / "nanogpt" / "final_metrics.json").exists():
        report["nanogpt_baseline"] = read_json(run / "nanogpt" / "final_metrics.json")
        report["nanogpt_generation"] = read_json(run / "nanogpt" / "generation.json")
    write_json(run / "evaluation" / "comparison.json", report)
    write_jsonl(run / "evaluation" / "generations.jsonl", samples)
    lines = ["# Model stage comparison", "", report["limits"], "", "| Stage | Raw loss | Perplexity | SFT response loss | Domain response loss | Chosen win rate | Cache equal |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['stage']} | {row['raw_validation_loss']:.4f} | {row['raw_perplexity']:.2f} | {row['sft_completion_loss']:.4f} | {row['domain_completion_loss']:.4f} | {row['preference_chosen_win_rate']:.1%} | {row['cache_greedy_equal']} |")
    if "nanogpt_baseline" in report:
        nano = report["nanogpt_baseline"]
        lines.append(f"| nanoGPT baseline | {nano['raw_validation_loss']:.4f} | {nano['perplexity']:.2f} | — | — | — | N/A |")
    lines += ["", "## Greedy generations", ""]
    for row in samples:
        lines += [f"### {row['stage']}: {row['prompt']}", "", "```text", row["text"], "```", ""]
    (run / "evaluation" / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report
