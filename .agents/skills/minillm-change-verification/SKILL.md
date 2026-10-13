---
name: minillm-change-verification
description: Verify MiniLLM code, configuration or dependency changes with the shared verifier and evidence tied to the current source snapshot.
---

Select checks according to the change, using the repository's existing implementation:

- Pure data transformations/tooling: `scripts/verify.ps1 -Profile Fast`.
- Runtime/API/checkpoint/dependency changes: `scripts/verify.ps1 -Profile CPU`.
- Training/model/precision/cache changes: CPU plus `scripts/verify.ps1 -Profile GPU` when CUDA is available.
- On non-PowerShell hosts use `python scripts/verify.py --profile fast|cpu|gpu` in the prepared environment.

Each invocation writes a fresh `reports/verification/<id>/summary.json` and individual logs. Use the
returned exit code and summary; do not treat archived reports as proof. A source change after verification
needs relevant checks again. Fix failures at their cause; changed acceptance criteria need an explanation.

Fixtures are temporary and self-contained. Keep pinned vendor repositories and historical runs intact.
For data/model changes also read `docs/engineering.md`'s change matrix; useful-model quality needs a
separate experiment review. Report profile, source identity, pass/fail and skipped checks. Verification
does not authorize merge or release.
