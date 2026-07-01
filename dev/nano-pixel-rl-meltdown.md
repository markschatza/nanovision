# Development Log

Record benchmark-changing decisions, smoke results, and long training session summaries here.

## 2026-07-01 - Aesop Signs-of-Life Attempts

Remote target: Aesop Ubuntu Linux with NVIDIA GeForce GTX 1660 Ti and JAX CUDA backend.

Validation before long runs:

- Remote `pytest`: `20 passed`.
- Remote smoke artifacts validated with `scripts/report.py`.

Counted completed sessions:

| Session | Remote artifact | Update time | Random/legal WR | Delayed WR | Threshold |
|---|---|---:|---:|---:|---|
| 1 | `artifacts/runs/aesop-session-001-2h/result.json` | `7200.20s` | `0.2421875` | `0.0` | `false` |
| 2 | `artifacts/runs/aesop-session-002-2h/result.json` | `7200.27s` | `0.12109375` | `0.0` | `false` |
| 3 | `artifacts/runs/aesop-session-003-2h/result.json` | `7200.05s` | `0.05078125` | `0.0` | `false` |

Stopped after the third complete validated session did not reach signs of life.
Failed starts and the interrupted original five-update-hour attempt were not counted.
