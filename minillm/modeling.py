"""Small integration layer over the pinned, unmodified MiniMind source."""
import importlib.util
import sys
from pathlib import Path

from torch import nn
from .common import ROOT, read_json

SOURCE = ROOT / "vendor" / "minimind" / "model" / "model_minimind.py"
spec = importlib.util.spec_from_file_location("lab_upstream_minimind", SOURCE)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
MiniMindConfig = upstream.MiniMindConfig


class LabMiniMind(upstream.MiniMindForCausalLM):
    # Transformers 4.x expects a list here; upstream also targets Transformers 5.
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False  # Upstream CE is a mean, not a num_items_in_batch sum.

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, upstream.RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, past_key_values[0][0].shape[1]:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask, "use_cache": kwargs.get("use_cache", True)}


def fresh_model(cfg, tokenizer):
    args = dict(cfg["model"])
    h, heads, kv = args["hidden_size"], args["num_attention_heads"], args["num_key_value_heads"]
    if h % heads or heads % kv or (h // heads) % 2:
        raise ValueError("Invalid attention dimensions or odd RoPE head dimension.")
    config = MiniMindConfig(**args, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                          bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                          use_moe=False, use_cache=False, initializer_range=0.02)
    return LabMiniMind(config)


def tokenizer_at(run):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(run / "tokenizer", local_files_only=True)


def load_model(path, device="cpu"):
    path = Path(path).resolve()
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        meta = read_json(path / "lineage.json")
        from .common import sha256
        parent = path.parent / meta["parent"]
        if sha256(parent / "model.safetensors") != meta["parent_weights_sha256"]:
            raise ValueError("LoRA parent weights changed; adapter lineage does not match.")
        base = load_model(parent, device)
        return PeftModel.from_pretrained(base, path, local_files_only=True).to(device)
    config = MiniMindConfig.from_pretrained(path, local_files_only=True)
    return LabMiniMind.from_pretrained(path, config=config, local_files_only=True).to(device)


def weight_stats(model):
    rows = []
    for name, p in model.named_parameters():
        x = p.detach().float()
        rows.append({"name": name, "shape": list(x.shape), "count": x.numel(), "mean": x.mean().item(), "std": x.std(unbiased=False).item(), "trainable": p.requires_grad})
    return rows
