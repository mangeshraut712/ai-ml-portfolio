# LLM Evaluation Lab Report

- Corpus docs: 10
- Gold queries: 40
- Adversarial queries: 15

## Retrieval (TF-IDF)
- Recall@5: 1.000
- MRR: 0.981
- nDCG@5: 0.986

## Rerank (BM25 fusion)
- Recall@3: 0.975
- MRR: 0.963
- nDCG@3: 0.964

## Prompt A/B
- concise: F1=0.341 faithfulness=0.850 p95=14.0ms
- verbose: F1=0.331 faithfulness=0.824 p95=14.0ms

## Hallucination rate (adversarial): 0.000

## Model comparison

| Provider | F1 | Faithfulness | p95 ms | $/1k q |
|---|---:|---:|---:|---:|
| openai | 0.351 | 0.861 | 12.0 | 0.0213 |
| anthropic | 0.345 | 0.817 | 18.0 | 0.1217 |
| google | 0.351 | 0.861 | 10.0 | 0.0142 |
| qwen | 0.345 | 0.817 | 15.0 | 0.0531 |
| gemma | 0.351 | 0.861 | 8.0 | 0.0000 |
| deepseek | 0.337 | 0.793 | 11.0 | 0.0172 |
| sarvam | 0.341 | 0.850 | 14.0 | 0.0665 |
