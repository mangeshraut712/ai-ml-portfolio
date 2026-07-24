# LLM Eval Gold Data Card

## Summary

| Split | Path | Size | Purpose |
|---|---|---:|---|
| QA gold | `data/gold/qa.jsonl` | **40** | Answerable Indic / RAG / speech / metrics queries |
| Adversarial | `data/gold/adversarial.jsonl` | **15** | Unanswerable hallucination traps |
| Corpus | `data/corpus/indic_ai_faq.md` | 10 FAQ sections (`doc_0`…`doc_9`) | Retrieval evidence |

## Schema

**QA / answerable**

```json
{"id": "qN", "question": "...", "answer": "...", "relevant_doc_ids": ["doc_k", ...]}
```

**Adversarial / unanswerable**

```json
{"id": "aN", "question": "...", "answer": "I don't know", "relevant_doc_ids": [], "unanswerable": true}
```

## Topics covered

- **Indic / Sarvam-style**: Sarvam overview, Bulbul TTS, Saaras STT, India-first positioning
- **Multilingual NLP**: code-mixing, dialects, script diversity, per-language eval
- **RAG systems**: faithfulness, abstention, first-stage retrieval, reranking
- **Eval metrics**: Recall@k, MRR, nDCG, embedding cosine scoring
- **Speech / latency**: interactive STT p95 &lt; 800 ms budgets
- **Production KPIs**: p50/p95 latency, $/1k queries
- **Interview pivots**: gradient descent, regularization, bias–variance, ROC-AUC, trees

## Labeling notes

- Answers are short, extractive paraphrases of FAQ sections (not free-form open-domain).
- `relevant_doc_ids` were assigned by mapping each question to the FAQ heading(s) that contain the claim; multi-doc ids used when abstention + faithfulness both apply.
- Adversarial items intentionally request facts **absent** from the corpus (menus, salaries, sports, medical advice, secrets). Gold answer is always `I don't know`.
- No PII; synthetic interview-style questions only.
- Paraphrase pairs (e.g. q1/q11) stress retrieval robustness without expanding the corpus.

## Metrics reported by the lab

| Metric | Gold used |
|---|---|
| Recall@k / MRR / nDCG@k | QA `relevant_doc_ids` |
| Token F1 (answer quality) | QA `answer` |
| Faithfulness | QA generations vs retrieved context |
| Hallucination rate | Adversarial (non-abstention = fail) |
| p50/p95 latency, $/1k | Both (token estimates × `pricing.yaml`) |

## Offline vs live

- Default / CI: `EVAL_LIVE` unset → stub generators (differentiated per provider style).
- Optional live: `EVAL_LIVE=1` + provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `SARVAM_API_KEY`, …). Missing keys stay stub; failures fall back to stub so the harness does not crash.
- See `configs/models.yaml` and `README.md`.

## Version

- Gold size: 40 QA + 15 adversarial (expanded from ~20 / 8).
- Last updated: 2026-07-24.
