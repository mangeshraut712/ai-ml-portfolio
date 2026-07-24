#!/usr/bin/env python3
"""Run the full LLM Evaluation Lab benchmark suite (offline-first)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metrics.answer_quality import token_f1  # noqa: E402
from eval_metrics.faithfulness import (  # noqa: E402
    faithfulness_score,
    hallucination_on_unanswerable,
)
from eval_metrics.latency_cost import estimate_cost_usd, percentile  # noqa: E402
from eval_metrics.retrieval import (  # noqa: E402
    aggregate_retrieval,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from generate import generate_answer  # noqa: E402
from ingest import load_corpus  # noqa: E402
from paths import CONFIG_DIR, CORPUS_DIR, GOLD_DIR, REPORT_DIR  # noqa: E402
from rerank import BM25Reranker  # noqa: E402
from retrieve import TfidfRetriever  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    corpus_path = CORPUS_DIR / "indic_ai_faq.md"
    docs = load_corpus(corpus_path)
    retriever = TfidfRetriever(docs)
    reranker = BM25Reranker(docs)

    models_cfg = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())
    pricing = yaml.safe_load((CONFIG_DIR / "pricing.yaml").read_text())
    exp = yaml.safe_load((CONFIG_DIR / "experiments" / "baseline_rag.yaml").read_text())

    gold = load_jsonl(GOLD_DIR / "qa.jsonl")
    adv = load_jsonl(GOLD_DIR / "adversarial.jsonl")
    top_k = int(exp.get("top_k", 5))
    rerank_k = int(exp.get("rerank_k", 3))

    # --- Retrieval + rerank ---
    retrieval_rows = []
    rerank_rows = []
    for item in gold:
        hits = retriever.search(item["question"], top_k=top_k)
        retrieved_ids = [h.doc_id for h in hits]
        rel = item["relevant_doc_ids"]
        retrieval_rows.append(
            {
                "id": item["id"],
                "recall_at_k": recall_at_k(rel, retrieved_ids, top_k),
                "mrr": mrr(rel, retrieved_ids),
                "ndcg_at_k": ndcg_at_k(rel, retrieved_ids, top_k),
            }
        )
        reranked = reranker.rerank(item["question"], hits, top_k=rerank_k)
        rr_ids = [h.doc_id for h in reranked]
        rerank_rows.append(
            {
                "id": item["id"],
                "recall_at_k": recall_at_k(rel, rr_ids, rerank_k),
                "mrr": mrr(rel, rr_ids),
                "ndcg_at_k": ndcg_at_k(rel, rr_ids, rerank_k),
            }
        )

    retrieval_summary = aggregate_retrieval(retrieval_rows, k=top_k)
    rerank_summary = aggregate_retrieval(rerank_rows, k=rerank_k)

    # --- Prompt A/B + faithfulness on gold ---
    prompt_results = {}
    for prompt_name, template in exp["prompts"].items():
        scores = []
        faith = []
        latencies = []
        for item in gold:
            hits = retriever.search(item["question"], top_k=top_k)
            reranked = reranker.rerank(item["question"], hits, top_k=rerank_k)
            contexts = [h.text for h in reranked]
            prompt = template.format(
                context="\n\n".join(contexts), question=item["question"]
            )
            gen = generate_answer(
                provider="sarvam",
                model="stub-sarvam",
                prompt=prompt,
                question=item["question"],
                contexts=contexts,
                style="verbose" if prompt_name == "verbose" else "concise",
            )
            scores.append(token_f1(gen.answer, item["answer"]))
            faith.append(faithfulness_score(gen.answer, contexts))
            latencies.append(gen.latency_ms)
        prompt_results[prompt_name] = {
            "mean_token_f1": sum(scores) / len(scores),
            "mean_faithfulness": sum(faith) / len(faith),
            "p95_latency_ms": percentile(latencies, 95),
        }

    # --- Hallucination on adversarial ---
    hallu = []
    for item in adv:
        hits = retriever.search(item["question"], top_k=top_k)
        contexts = [h.text for h in hits[:rerank_k]]
        gen = generate_answer(
            provider="sarvam",
            model="stub-sarvam",
            prompt=item["question"],
            question=item["question"],
            contexts=contexts,
            unanswerable=True,
        )
        hallu.append(hallucination_on_unanswerable(gen.answer))
    hallucination_rate = sum(1 for h in hallu if h) / max(len(hallu), 1)

    # --- Model comparison (stubbed providers) ---
    comparison = []
    for provider in models_cfg["comparison_models"]:
        meta = models_cfg["providers"][provider]
        latencies = []
        faith_scores = []
        f1_scores = []
        modes = []
        in_tok = 0
        out_tok = 0
        t_batch = time.perf_counter()
        for item in gold:
            hits = retriever.search(item["question"], top_k=top_k)
            reranked = reranker.rerank(item["question"], hits, top_k=rerank_k)
            contexts = [h.text for h in reranked]
            prompt = exp["prompts"]["concise"].format(
                context="\n\n".join(contexts), question=item["question"]
            )
            gen = generate_answer(
                provider=provider,
                model=meta["default_model"],
                prompt=prompt,
                question=item["question"],
                contexts=contexts,
                style=meta.get("stub_style", "concise"),
            )
            latencies.append(gen.latency_ms)
            faith_scores.append(faithfulness_score(gen.answer, contexts))
            f1_scores.append(token_f1(gen.answer, item["answer"]))
            modes.append(gen.mode)
            in_tok += gen.prompt_tokens
            out_tok += gen.completion_tokens
        wall_ms = (time.perf_counter() - t_batch) * 1000
        price = pricing["pricing_per_mtok"][provider]
        n = len(gold)
        cost = estimate_cost_usd(
            n_queries=n,
            input_tokens=in_tok / n,
            output_tokens=out_tok / n,
            price_in_per_mtok=price["input"],
            price_out_per_mtok=price["output"],
        )
        comparison.append(
            {
                "provider": provider,
                "model": meta["default_model"],
                "mean_token_f1": round(sum(f1_scores) / n, 4),
                "mean_faithfulness": round(sum(faith_scores) / n, 4),
                "p50_latency_ms": round(percentile(latencies, 50), 2),
                "p95_latency_ms": round(percentile(latencies, 95), 2),
                "batch_wall_ms": round(wall_ms, 2),
                "est_cost_usd_for_gold_set": round(cost, 6),
                "est_cost_usd_per_1k_queries": round(cost / n * 1000, 4),
                "mode": ("live" if modes and all(m == "live" for m in modes) else ("mixed" if any(m == "live" for m in modes) else "stub")),
            }
        )

    report = {
        "corpus_docs": len(docs),
        "gold_n": len(gold),
        "adversarial_n": len(adv),
        "retrieval": retrieval_summary,
        "rerank": rerank_summary,
        "prompt_ab": prompt_results,
        "hallucination_rate_adversarial": hallucination_rate,
        "model_comparison": comparison,
        "notes": [
            "Offline stubs by default; set EVAL_LIVE=1 with provider API keys for live calls.",
            "TF-IDF is the local embedding baseline; swap for OpenAI/Sarvam embeddings later.",
            "BM25 fusion reranks first-stage hits.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "latest_report.json"
    out_md = REPORT_DIR / "latest_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# LLM Evaluation Lab Report",
        "",
        f"- Corpus docs: {len(docs)}",
        f"- Gold queries: {len(gold)}",
        f"- Adversarial queries: {len(adv)}",
        "",
        "## Retrieval (TF-IDF)",
        f"- Recall@{top_k}: {retrieval_summary['recall_at_k']:.3f}",
        f"- MRR: {retrieval_summary['mrr']:.3f}",
        f"- nDCG@{top_k}: {retrieval_summary['ndcg_at_k']:.3f}",
        "",
        "## Rerank (BM25 fusion)",
        f"- Recall@{rerank_k}: {rerank_summary['recall_at_k']:.3f}",
        f"- MRR: {rerank_summary['mrr']:.3f}",
        f"- nDCG@{rerank_k}: {rerank_summary['ndcg_at_k']:.3f}",
        "",
        "## Prompt A/B",
    ]
    for name, block in prompt_results.items():
        lines.append(
            f"- {name}: F1={block['mean_token_f1']:.3f} "
            f"faithfulness={block['mean_faithfulness']:.3f} "
            f"p95={block['p95_latency_ms']:.1f}ms"
        )
    lines += [
        "",
        f"## Hallucination rate (adversarial): {hallucination_rate:.3f}",
        "",
        "## Model comparison",
        "",
        "| Provider | F1 | Faithfulness | p95 ms | $/1k q |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['provider']} | {row['mean_token_f1']:.3f} | "
            f"{row['mean_faithfulness']:.3f} | {row['p95_latency_ms']:.1f} | "
            f"{row['est_cost_usd_per_1k_queries']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(out_md.read_text())
    print(f"[+] Wrote {out_json}")
    print(f"[+] Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
