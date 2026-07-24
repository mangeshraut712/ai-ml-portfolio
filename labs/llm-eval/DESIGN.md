# LLM Evaluation Lab — Design

## Problem
Product demos show *that* an LLM can answer. Hiring bars ask *how well*, at what
latency/cost, and whether answers are grounded.

## Pipeline
```text
corpus → TF-IDF index → top-k retrieve → BM25 fusion rerank → prompt → answer
                                              ↓
                         Recall@k / MRR / nDCG / faithfulness / $/1k
```

## Why TF-IDF first
Offline, deterministic, no API keys — CI always green. Swap for dense embeddings
(OpenAI / Sarvam / local sentence-transformers) when `EVAL_LIVE=1`.

## Failure modes we measure
| Failure | Metric |
|---|---|
| Missed relevant doc | Recall@k, MRR |
| Wrong ranking | nDCG@k |
| Ungrounded answer | Faithfulness lexical support |
| Fabrication on unanswerable | Hallucination rate |
| Slow / expensive | p95 latency, $/1k queries |

## Tradeoffs
- Larger `top_k` raises recall, hurts latency and distractor context.
- Strict abstention prompts lower hallucination, may lower answer F1.
- Stub generators equalize model rows offline; live keys differentiate quality.
