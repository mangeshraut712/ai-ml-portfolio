# LLM Eval Results

**Date:** 2026-07-24  
**Mode:** **STUB** (CI-safe) — no live provider keys present (`OPENAI_API_KEY` / `SARVAM_API_KEY` / `ANTHROPIC_API_KEY` unset).  
**CI:** always runs stub (`EVAL_LIVE` unset). Live path is optional and local-only.

Gold: 40 QA + 15 adversarial · corpus: 10 FAQ docs — see [`DATA_CARD.md`](DATA_CARD.md).

## How to reproduce

```bash
# Stub baseline (CI-equivalent)
make lab-llm-eval

# Optional live providers (requires keys; not used in CI)
EVAL_LIVE=1 make lab-llm-eval-live
```

Artifacts: `reports/latest_report.json`, `reports/latest_report.md`.

## Retrieval (TF-IDF) — STUB

| Metric | Value |
|---|---:|
| Recall@5 | **1.000** |
| MRR | **0.981** |
| nDCG@5 | **0.986** |

## Rerank (BM25 fusion) — STUB

| Metric | Value |
|---|---:|
| Recall@3 | **0.975** |
| MRR | **0.963** |
| nDCG@3 | **0.964** |

## Prompt A/B — STUB

| Variant | Token F1 | Faithfulness | p95 latency |
|---|---:|---:|---:|
| concise | 0.341 | 0.850 | ~14 ms |
| verbose | 0.331 | 0.824 | ~14 ms |

## Hallucination (adversarial) — STUB

| Metric | Value |
|---|---:|
| Hallucination rate | **0.000** |

## Model comparison — STUB

Provider-flavored offline stubs (differentiated styles; not live API scores).

| Provider | Token F1 | Faithfulness | p95 ms | $/1k q | Mode |
|---|---:|---:|---:|---:|---|
| openai | 0.351 | 0.861 | 12.0 | 0.0213 | stub |
| anthropic | 0.345 | 0.817 | 18.0 | 0.1217 | stub |
| google | 0.351 | 0.861 | 10.0 | 0.0142 | stub |
| qwen | 0.345 | 0.817 | 15.0 | 0.0531 | stub |
| gemma | 0.351 | 0.861 | 8.0 | 0.0000 | stub |
| deepseek | 0.337 | 0.793 | 11.0 | 0.0172 | stub |
| sarvam | 0.341 | 0.850 | 14.0 | 0.0665 | stub |

## Live eval

Not run on 2026-07-24 — no API keys in environment. To capture live metrics:

```bash
EVAL_LIVE=1 OPENAI_API_KEY=… make lab-llm-eval-live
# then refresh this file from reports/latest_report.md (mode=live / mixed)
```

Missing keys fall back to stub so the harness does not crash; CI must never set `EVAL_LIVE`.
