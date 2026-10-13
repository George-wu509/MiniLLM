# MiniLLM engineering contract

MiniLLM is a Windows laptop training laboratory: MiniMind/nanoGPT from random weights, TRL SFT/DPO,
and PEFT domain LoRA. Keep that teaching behavior explicit. Runtime code is in `minillm/`.

## Development and verification

- Use a task branch; concurrent writers use separate worktrees. Do not overwrite another task's changes.
- Bootstrap with `scripts/setup.ps1 -Development` (CUDA) or add `-CpuOnly`. Upstream commits are in
  `upstream-lock.json`; keep vendor checkouts unmodified and put integration fixes in `minillm/`.
- For runtime/config/dependency changes use `$minillm-change-verification` and the shared verifier:
  `scripts/verify.ps1 -Profile CPU` or `python scripts/verify.py --profile cpu`.
  Fast checks suit pure data/tooling changes; training/model/cache changes also need GPU evidence
  when CUDA is available. Explain unavailable checks; do not silently substitute CPU for GPU.
- Tests build their own synthetic fixtures. Never require or edit historical `artifacts/smoke`.
- Report current evidence path, commit/source hash, actual commands and limitations. A historical
  report, AI review or a numeric metric is not current verification or human release approval.
- Execute authorized local edits, temporary fixtures and bounded validation autonomously. Existing
  task authorization persists; do not ask for each routine operation.

## Training invariants

- Keep train/validation prompt grouping, train-only tokenizer training, causal masking, prompt/pad
  loss masking, accumulation normalization and LoRA frozen-parent behavior intact.
- Data/tokenizer/loss/context changes need new run IDs and relevant regression tests. Retain completed
  runs; compare compatible protocols with `$minillm-experiment-review`.
- Resume must preserve dataset/tokenizer/config/parent identity. Only load trusted training states;
  native optimizer checkpoints use pickle. Inference candidates exclude optimizer/RNG `.pt` files.
- Use synthetic data for verification; long training and external/private data require the task's
  resource/data authorization. The GPU verifier runs only two steps per stage in temporary storage.

## Collaboration and release

- PRs state acceptance criteria, validation evidence, compatibility and rollback. Review changes to
  CI, permissions, tests, locks and these instructions as carefully as runtime changes.
- Skills guide decisions; scripts/tests/CI and external GitHub/OS policies enforce boundaries.
- Build/check candidates with `scripts/release.py`; candidates remain `candidate-not-approved`.
  Preserve model/tokenizer/evaluation hashes. Serving uses localhost and logs outside immutable bundles.
- Never infer permission to merge, publish, deploy, delete baseline models or widen credentials from
  passing tests. Follow explicit user authorization and configured approval boundaries.
- See `docs/engineering.md` for setup/SDLC examples and `docs/release.md` for promotion/rollback.

## Code review rules

Check data leakage, all-masked completions, length handling, optimizer-step semantics, resume
provenance, adapter-parent compatibility and cache equivalence. Distinguish structural smoke checks
from useful-model/task-quality evidence. Flag changes that weaken verification or release controls.
