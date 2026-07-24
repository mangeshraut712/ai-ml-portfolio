# LLM Evaluation Lab Report

- Corpus docs: 10
- Gold queries: 20
- Adversarial queries: 8

## Retrieval (TF-IDF)
- Recall@5: 1.000
- MRR: 0.963
- nDCG@5: 0.972

## Rerank (BM25 fusion)
- Recall@3: 0.950
- MRR: 0.950
- nDCG@3: 0.946

## Prompt A/B
- concise: F1=0.418 faithfulness=0.891 p95=14.0ms
- verbose: F1=0.403 faithfulness=0.833 p95=14.0ms

## Hallucination rate (adversarial): 0.000

## Model comparison (stub)

| Provider | F1 | Faithfulness | p95 ms | $/1k q |
|---|---:|---:|---:|---:|
| openai | 0.418 | 0.891 | 12.0 | 0.0210 |
| anthropic | 0.418 | 0.891 | 18.0 | 0.1192 |
| google | 0.418 | 0.891 | 10.0 | 0.0140 |
| qwen | 0.418 | 0.891 | 15.0 | 0.0524 |
| gemma | 0.418 | 0.891 | 8.0 | 0.0000 |
| deepseek | 0.418 | 0.891 | 11.0 | 0.0171 |
| sarvam | 0.418 | 0.891 | 14.0 | 0.0655 |
