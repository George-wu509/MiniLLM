---
name: minillm-experiment-review
description: Compare MiniLLM baseline and candidate experiments when data, tokenizer, context, losses or training settings change, preserving lineage and evaluation comparability.
---

Read `docs/release.md` and inspect the requested runs' config, data manifest, stage lineage and evaluation.
Establish code commit/source hash, tokenizer identity, dataset split, parent weights, evaluation protocol
and precision before comparing metrics. Different tokenizers change loss/perplexity units; different
datasets or preference objectives are not interchangeable. Report such differences instead of ranking
incompatible numbers. Use task accuracy or a separate held-out suite when usefulness is the acceptance goal.

Never update completed runs in place. New training uses new run IDs within the authorized data/resource
budget. The bounded verifier may validate mechanics; long training is a separate experiment.

If packaging was requested, use `scripts/release.py bundle` and `check` with an explicit policy. Merge
LoRA before packaging a full model. Historical runs missing MiniLLM provenance need a new traced run;
do not fabricate identity. A candidate manifest records integrity, not approval. Present baseline/candidate
differences, evidence, risks and rollback version for the human release decision.
