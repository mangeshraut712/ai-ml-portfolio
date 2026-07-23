# AI/ML Portfolio Map

Interview-ready signal map for applied MLE / speech / Indic AI / research-leaning roles.

## Role fit (honest)

| Role type | Fit | Why |
|---|---|---|
| Applied ML Engineer | **Strong** | Shipping apps + VAD systems + RAG eval harness |
| Speech / Audio ML | **Strong** | Proctored-style VAD with labeled F1 + latency gates |
| LLM / RAG Engineer | **Good → Strong** | Retrieval/faithfulness/cost lab (expand gold + live providers for research bar) |
| Classical ML interview pivot | **Strong** | NumPy-from-scratch + math notes |
| Research Scientist (DL-heavy) | **Emerging** | MLP + attention from scratch added; still needs a public train/eval card |

## Portfolio map

| Repo | Primary signal | Prove in 60 seconds | Interview line |
|---|---|---|---|
| [`sarvam-ai-cookbook`](https://github.com/mangeshraut712/sarvam-ai-cookbook) | **Ship + evaluate** India-first AI | `make verify-vad-pass` · `make lab-llm-eval` | “I build production-shaped speech/LLM systems and measure them.” |
| [`machine-learning-from-scratch`](https://github.com/mangeshraut712/machine-learning-from-scratch) | **Defend fundamentals** (classical + mini-DL) | `make test && make demo` | “I can derive GD, regularization, trees, ROC, backprop, and attention.” |
| [`ai-ml-portfolio`](https://github.com/mangeshraut712/ai-ml-portfolio) *(this repo)* | **Recruiter index** | Read this README | “Here is the map of what I own and how to verify it.” |

```text
                    ┌─────────────────────────┐
                    │   ai-ml-portfolio       │
                    │   (index / narrative)   │
                    └───────────┬─────────────┘
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────────┐
│ sarvam-ai-cookbook          │     │ machine-learning-from-scratch   │
│ • examples/ (apps)          │     │ • classical ML (NumPy)          │
│ • VAD challenge (FULL_PASS) │     │ • MLP + attention (from scratch)│
│ • lab/llm-eval (RAG/metrics)│     │ • bias–variance, ROC, CV        │
└─────────────────────────────┘     └─────────────────────────────────┘
```

## What “qualified” means here

A repo qualifies an AI/ML candidate when a reviewer can:

1. **Clone → run one command → see measured results** (not screenshots only)
2. Hear a **defendable trade-off** (accuracy vs latency, stub vs live, L1 vs L2)
3. See **tests / gates** that fail when quality regresses
4. Map work to a **job function** (systems, eval, fundamentals)

## Verification cheat sheet

```bash
# Applied systems + eval
cd sarvam-ai-cookbook
make verify-vad-pass
make lab-llm-eval
make lab-llm-eval-test

# Fundamentals
cd machine-learning-from-scratch
make install && make test && make demo
```

## Target roles & talking points

**MLE / Applied AI**
- VAD: Denoiser → WebRTC GMM → hangover; clean exact F1 ≈ 0.96; 258/258 frame coverage; p95 ≈ 20 ms
- LLM lab: Recall@k / MRR / nDCG, faithfulness, hallucination rate, $/1k queries, multi-model matrix

**When they pivot off LLMs**
- Derive `θ ← θ − η∇L`, L1 sparsity vs L2 shrinkage, bias–variance U-curve, ROC-AUC meaning
- Sketch backprop through an MLP and scaled-dot-product attention

## Roadmap (next hiring-signal upgrades)

- [ ] Live provider runs in `lab/llm-eval` (`EVAL_LIVE=1`) with differentiated scores
- [ ] Expand gold set to 50–100+ queries + data card
- [ ] Neural VAD A/B (ONNX) next to GMM in the cookbook
- [ ] Public fine-tune + eval card for research-track interviews
