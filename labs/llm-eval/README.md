# LLM Evaluation Lab

Interview-grade **benchmarking / experimentation / evaluation** harness inside the Sarvam AI Cookbook.

Closes the portfolio gap the VAD challenge does not cover: RAG retrieval quality,
faithfulness, hallucination traps, latency/cost, prompt A/B, and multi-model comparison.

## Why this lives here

Yes — this cookbook is a valid home. Keep product demos under `examples/` and put
systematic evaluation under `lab/llm-eval/` so recruiters can find both:

| Path | Story |
|---|---|
| `examples/` | Ship India-first apps |
| `lab/llm-eval/` | Measure models like a research / applied-AI engineer |
| `examples/sarvam-vad-challenge/` | Speech ML under interview constraints |

Classical ML fundamentals live in this monorepo under [`mlfs/`](../../mlfs/).
Sarvam product demos: [`sarvam-ai-cookbook`](https://github.com/mangeshraut712/sarvam-ai-cookbook).

## One-command demo (offline, no API keys)

```bash
# from repo root
.venv/bin/pip install -q scikit-learn rank-bm25 pyyaml
make lab-llm-eval
```

This runs:

1. Ingest FAQ corpus → TF-IDF (+ optional dense) index  
2. Retrieval metrics: Recall@k, MRR, nDCG  
3. Rerank fusion  
4. Faithfulness / hallucination heuristics on gold + adversarial sets  
5. Prompt A/B  
6. Latency + cost estimate table  
7. Model comparison leaderboard (configured providers; offline stubs if no keys)

## Layout

```text
lab/llm-eval/
├── configs/models.yaml pricing.yaml experiments/
├── data/corpus/  data/gold/qa.jsonl  data/gold/adversarial.jsonl
├── src/          ingest retrieve rerank generate compare report + metrics/
├── tests/
├── notebooks/llm_eval_walkthrough.ipynb
├── ui/streamlit_app.py
└── reports/      # generated JSON/MD
```

## Metrics glossary (say these in interviews)

| Metric | Measures |
|---|---|
| Recall@k | Fraction of queries where a relevant doc is in top-k |
| MRR | Mean reciprocal rank of first relevant hit |
| nDCG@k | Rank-discounted graded relevance |
| Faithfulness | Answer claims supported by retrieved context |
| Hallucination rate | Answers invented on unanswerable / adversarial queries |
| p50/p95 latency | End-to-end retrieve+generate wall time |
| $/1k queries | Token pricing × usage |

## Model comparison matrix

Configured in `configs/models.yaml`:

OpenAI · Claude · Gemini · Qwen · Gemma · DeepSeek · Sarvam

Runs offline with deterministic stub generators when `API` keys are absent so CI
and local demos always work. Plug real keys via env vars to swap stubs for live calls.
