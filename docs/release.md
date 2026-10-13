# Experiment tracking, inference candidates and rollback

## Identity and comparison

New stage lineage includes MiniLLM Git commit, dirty flag and source SHA256, package versions,
Python/OS, upstream commits and dependency lock hashes. An uncommitted experiment is identifiable
but is not a reviewed release commit. Existing historical metadata is preserved, not rewritten.

Prepared data is rehashed before initialization/training/resume. New tokenizer manifests cover
tokenizer.json plus config/template files. Old manifests retain their original tokenizer.json-only
coverage. hashes detect changed bytes; they do not authenticate an attacker who can rewrite metadata.

Evaluation now binds the weights, tokenizer, dataset manifest and evaluation source identity.
Compare baseline/candidate only after confirming identical validation/tokenizer/protocol units:

```powershell
.\.venv\Scripts\python.exe scripts/compare_runs.py --baseline issue23-baseline --candidate issue23-candidate --stage sft --output reports/verification/issue23-comparison.json
```

This command refuses missing/stale weight identity, differing tokenizer/validation/windows/library
versions. It does not create fresh evaluation silently or rank incompatible metrics. Historical
evaluations can be explicitly re-evaluated in a copy of their run if preservation is required.
Source changes, parent lineage and task accuracy still need contextual review.

## Build and check a candidate

Create a new traced experiment using a fresh run ID; do not retrain in artifacts/smoke:

```powershell
.\scripts\lab.ps1 all --run release-demo-01 --config configs/ci.json
# ci.json uses CPU and two steps, only for demonstrating the pipeline.
.\.venv\Scripts\python.exe scripts/release.py bundle --run release-demo-01 --stage sft --destination releases/release-demo-01
.\.venv\Scripts\python.exe scripts/release.py check --bundle releases/release-demo-01 --policy configs/release-policy.lab.json
```

The package contains full weights, model/run config, tokenizer, data manifest, lineage and exact stage
evaluation. Optimizer/RNG checkpoints and raw corpus are excluded. Candidate directory must be new,
outside the training run; it cannot be overwritten. For LoRA run `lab.ps1 merge` first, then package
the `merged` stage so serving has a standalone full model. Keep original adapter+parent in the experiment.

Validation rehashes all packaged files, rejects path escapes/symlinks, missing/extra files, stale
evaluation identity, nonfinite losses and failed cache consistency. A policy applies explicit numeric
metric bounds. `configs/release-policy.lab.json` is **laboratory-only**, with permissive sanity bounds.
It does not certify language quality. A real release needs a separately defined, independent held-out
task suite and acceptance policy; this repository's synthetic two-step model cannot establish usefulness.

The manifest always says `candidate-not-approved`. This tool never approves, uploads or deploys.
Historical models missing MiniLLM training provenance are refused: create a traced run rather than
fabricating a historical code commit. The full release ID hashes the manifest metadata and file hashes;
it is integrity evidence, not a digital signature.

## Inspect in a local service

After reviewing the candidate, launch a localhost-only service:

```powershell
.\.venv\Scripts\python.exe scripts/serve_release.py --bundle releases/release-demo-01 --policy configs/release-policy.lab.json --logs .cache/runtime/release-demo-01 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
```

`health` and request metadata include release_id. Runtime logs cannot be inside the bundle, so the
read-only weights can be protected with OS ACLs. API logs omit prompt/response contents. Existing
`lab.ps1 serve` remains available for experimental runs. Stop a foreground service with Ctrl+C.

## Promotion and rollback

Human release record should contain Issue/PR/approved code commit, release ID and immutable storage
location, current CPU/GPU evidence, independent task-quality results, approver/time, deployment target,
and previous release ID. Archive the accepted quality policy alongside the release decision.

For a real deployment use a separate service/deployment account. Developer Agents have no production
credentials or write access to accepted model storage. GitHub main rules and environment reviewers are
external settings; see engineering.md. The deployment job must verify the same accepted bytes rather
than retraining or resolving dependencies on the target. Service authentication/capacity monitoring
must match the actual deployment environment; this repo currently exposes a teaching localhost API.

On regression: record current release ID and failure signal; route traffic or restart the service with
the prior accepted **code+dependency environment+model+tokenizer** bundle; run health and acceptance
checks; open a new fix Issue. Git revert alone does not restore ignored model files. Do not delete or
rebuild the previous model while rolling back.
