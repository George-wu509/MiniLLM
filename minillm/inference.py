import time

import torch
from .common import seed_all, device_for
from .modeling import load_model, tokenizer_at


class Engine:
    def __init__(self, run, cfg, stage):
        seed_all(cfg["seed"])
        # Keep evaluation stable regardless of whether a previous HF Trainer enabled TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = device_for(cfg)
        self.tokenizer = tokenizer_at(run)
        self.model = load_model(run / stage, self.device).eval()
        self.stage = stage

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=48, temperature=0.8, top_p=0.9, top_k=0,
                 use_cache=True, greedy=False, chat=True, seed=42, stop_eos=True):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not 0 < temperature <= 5 or not 0 < top_p <= 1 or top_k < 0 or not 1 <= max_new_tokens <= 512:
            raise ValueError("Invalid temperature/top_p/top_k/max_new_tokens.")
        tokenizer = self.tokenizer
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if chat else prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Prompt has no tokens.")
        if len(ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus max_new_tokens exceeds model context capacity.")
        # Configure the global thread pool once at Engine construction;
        # requests only reset RNG, including when called from an API worker.
        seed_all(seed, configure_threads=False)
        input_ids = torch.tensor([ids], device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                     top_k=min(top_k, len(tokenizer)), use_cache=use_cache,
                                     do_sample=not greedy, eos_token_id=tokenizer.eos_token_id if stop_eos else None)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        tokens = output[0, len(ids):].tolist()
        return {"stage": self.stage, "prompt": prompt, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens, "prompt_tokens": len(ids), "completion_tokens": len(tokens),
                "seconds": seconds, "tokens_per_second": len(tokens)/max(seconds, 1e-9),
                "settings": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "use_cache": use_cache, "greedy": greedy, "chat": chat, "seed": seed}}


def cache_benchmark(engine, prompt="Explain SFT.", tokens=24):
    # Warm up before timing; force the same decode length for both configurations.
    engine.generate(prompt, max_new_tokens=2, greedy=True, stop_eos=False)
    results = [engine.generate(prompt, max_new_tokens=tokens, greedy=True, top_p=1,
                               temperature=1, use_cache=cache, stop_eos=False) for cache in (False, True)]
    return {"without_cache": results[0], "with_cache": results[1],
            "greedy_token_ids_equal": results[0]["token_ids"] == results[1]["token_ids"],
            "speedup": results[0]["seconds"] / results[1]["seconds"],
            "note": "Timing is hardware/workload dependent; tiny models may show little speedup."}
