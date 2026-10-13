from minillm.common import ROOT, read_json
from minillm.data import demo_data, prepare
from minillm.pretrain import initialize


def prepared_fixture(root):
    """Generate deterministic, public synthetic inputs in a caller-owned directory."""
    cfg = read_json(ROOT / "configs" / "ci.json")
    raw, run = root / "raw", root / "run"
    demo_data(raw)
    prepare(run, cfg, {kind: raw / f"{name}.jsonl" for kind, name in
                      (("pretrain", "pretrain"), ("sft", "sft"), ("dpo", "dpo"), ("domain", "domain"))})
    return run, cfg


def model_fixture(root):
    import shutil
    run, cfg = prepared_fixture(root)
    initialize(run, cfg)
    # Parent fixtures are deliberately random models. Tests train their own stages.
    for stage in ("base", "sft", "dpo"):
        shutil.copytree(run / "initialized", run / stage)
    return run, cfg
