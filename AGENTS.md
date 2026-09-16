# AGENTS.md — ai-ml-portfolio

Guidance for coding agents (Copilot, Codex, Claude Code, Cursor) working in this repo.

## Layout

- `mlfs/` — NumPy ML from scratch (no sklearn in implementations; sklearn only in tests/demos).
- `labs/vad/` — speech VAD pipeline with an acceptance gate. See `labs/vad/MODEL_CARD.md`.
- `labs/llm-eval/` — offline RAG / LLM evaluation harness. See `labs/llm-eval/DATA_CARD.md` and `RESULTS.md`.
- `tests/` — mlfs unit tests. Lab tests live next to each lab.

## Setup and verification

```bash
make install        # creates .venv and installs requirements.txt
make verify-all     # CI-equivalent: mlfs tests + VAD FULL_PASS gate + offline LLM eval
```

Python 3.10–3.12 (CI matrix). Always run through `.venv/bin/python` (the Makefile does).

## Hard rules

1. **CI must never call a live model API.** `make lab-llm-eval` runs stub providers. The live path is
   `EVAL_LIVE=1 make lab-llm-eval-live` and is local-only. Do not set `EVAL_LIVE` in workflows.
2. **Do not lower gates to make CI green.** Thresholds in `labs/vad/challenge_pass.py` (`GATES`) are the
   acceptance contract. If a change drops a metric, fix the change or document the regression.
3. **Keep the VAD gate deterministic.** WebRTC aggressiveness 2 is the FULL_PASS configuration; the
   neural/energy backends are A/B only. Dataset revision in `labs/vad/fetch_samples.py` stays pinned.
4. **Update the numbers where they live.** Metrics are reported in `README.md`, `labs/vad/MODEL_CARD.md`
   and `labs/llm-eval/RESULTS.md`. If you change a pipeline, rerun the gate and update all three from the
   generated report — never hand-edit a number.
5. Standard library + pinned `requirements.txt` only; no new heavy dependencies without a note in the PR.

## Useful commands

```bash
make test-mlfs            # NumPy ML tests
make verify-vad-pass      # VAD gate (fetches pinned samples if missing)
make lab-llm-eval-test    # eval harness smoke tests
make run-vad-ui           # Streamlit VAD timeline
make gallery              # refresh README screenshots
```
