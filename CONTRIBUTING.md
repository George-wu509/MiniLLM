# Contributing to MiniLLM

1. Define desired behavior, acceptance examples, data/resource budget and compatibility in an Issue.
2. Use a task branch; concurrent work uses a separate worktree. Bootstrap that checkout's environment.
3. Make a scoped change and meaningful behavior tests. Preserve the LLM lifecycle invariants in AGENTS.md.
4. Run `scripts/verify.ps1 -Profile CPU`; model/training changes also use GPU when available.
5. Open a PR using the template. Link current verification evidence; explain unavailable checks.
6. Resolve review findings. GitHub required checks and human review govern merging; AI comments do not approve.
7. For a service/model release follow docs/release.md and preserve the previous release for rollback.

Setup, Skills, GitHub settings and worked Codex/Claude SDLC prompts: [engineering guide](docs/engineering.md).
Production deployment remains an external, separately approved operation; this repo serves on localhost.
