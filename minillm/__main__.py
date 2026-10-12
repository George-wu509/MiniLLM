import argparse
import json
from pathlib import Path

from .common import ROOT, read_json, write_json, run_path


def main():
    parser = argparse.ArgumentParser(description="MiniMind + nanoGPT + TRL local LLM lifecycle laboratory")
    parser.add_argument("command", choices=["doctor", "demo-data", "prepare", "init", "pretrain", "sft", "dpo", "lora", "nanogpt", "evaluate", "infer", "cache", "merge", "serve", "all"])
    parser.add_argument("--run", default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--sft-data", type=Path)
    parser.add_argument("--dpo-data", type=Path)
    parser.add_argument("--domain-data", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--parent", choices=["base", "sft", "dpo"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", default="dpo", choices=["initialized", "base", "sft", "dpo", "lora", "merged"])
    parser.add_argument("--prompt", default="Calculate 2 + 3. Answer with a number.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "doctor":
        from .common import doctor
        doctor()
        return
    from .data import demo_data, prepare
    if args.command == "demo-data":
        demo_data(args.destination)
        print(args.destination)
        return
    run = run_path(args.run)
    if args.command in {"prepare", "all"}:
        cfg = read_json(args.config or ROOT / "configs" / "smoke.json")
        if args.command == "all" and not args.raw and not args.destination.exists():
            demo_data(args.destination)
        paths = {"pretrain": args.raw or args.destination / "pretrain.jsonl", "sft": args.sft_data or args.destination / "sft.jsonl", "dpo": args.dpo_data or args.destination / "dpo.jsonl", "domain": args.domain_data or args.destination / "domain.jsonl"}
        if args.command == "prepare" or not (run / "config.json").exists():
            prepare(run, cfg, paths)
        elif cfg != read_json(run / "config.json"):
            raise ValueError("Existing run has a different configuration. Choose a new --run.")
        if args.command == "prepare":
            return
    else:
        cfg = read_json(run / "config.json")
        if args.config and cfg != read_json(args.config):
            raise ValueError("Configuration must match the prepared run.")
    from .pretrain import initialize, train as pretrain
    from .posttrain import train as posttrain
    from .evaluate import compare
    from .inference import Engine, cache_benchmark
    if args.command == "all":
        if any((run / stage / file).exists() for stage in ("base", "sft", "dpo", "lora") for file in ("model.safetensors", "adapter_model.safetensors")):
            raise FileExistsError("A completed stage already exists. Run remaining individual stages or use a new run.")
        initialize(run, cfg)
        pretrain(run, cfg)
        posttrain(run, cfg, "sft")
        posttrain(run, cfg, "dpo")
        posttrain(run, cfg, "lora")
        compare(run, cfg)
    elif args.command == "init":
        initialize(run, cfg)
        compare(run, cfg, ["initialized"])
    elif args.command in {"pretrain", "nanogpt"}:
        pretrain(run, cfg, args.resume, nano=args.command == "nanogpt")
        if args.command == "pretrain":
            compare(run, cfg, ["base"])
    elif args.command in {"sft", "dpo", "lora"}:
        posttrain(run, cfg, args.command, args.parent, args.resume)
        compare(run, cfg, [args.command])
    elif args.command == "evaluate":
        compare(run, cfg)
    elif args.command == "infer":
        result = Engine(run, cfg, args.stage).generate(args.prompt, args.max_new_tokens, args.temperature, args.top_p,
                                                    args.top_k, not args.no_cache, args.greedy, not args.raw_prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        write_json(run / "evaluation" / "last_inference.json", result)
    elif args.command == "cache":
        result = cache_benchmark(Engine(run, cfg, args.stage), args.prompt)
        write_json(run / "evaluation" / f"{args.stage}_cache.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "merge":
        from .modeling import load_model, tokenizer_at
        target = run / "merged"
        if target.exists():
            raise FileExistsError("Merged stage already exists.")
        model = load_model(run / "lora")
        model.merge_and_unload().save_pretrained(target)
        tokenizer_at(run).save_pretrained(target)
        write_json(target / "lineage.json", {"parent": "lora", "method": "PEFT merge_and_unload"})
        compare(run, cfg, ["merged"])
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(run, cfg, args.stage), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
