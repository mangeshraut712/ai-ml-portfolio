"""Generators: offline stubs + optional live provider hooks."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class GenerationResult:
    provider: str
    model: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    mode: str  # stub | live


# Env vars checked when EVAL_LIVE=1
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemma": "GOOGLE_API_KEY",
    "sarvam": "SARVAM_API_KEY",
    "qwen": "OPENAI_API_KEY",  # OpenAI-compatible optional
    "deepseek": "DEEPSEEK_API_KEY",
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def live_enabled() -> bool:
    return os.environ.get("EVAL_LIVE", "0").strip() == "1"


def provider_api_key(provider: str) -> str | None:
    env_name = PROVIDER_ENV.get(provider)
    if not env_name:
        return None
    key = os.environ.get(env_name, "").strip()
    return key or None


def stub_answer(
    provider: str,
    question: str,
    contexts: list[str],
    style: str = "concise",
    force_abstain: bool = False,
) -> str:
    """Deterministic offline generator for CI / demos."""
    if force_abstain or not contexts:
        return "I don't know"
    blob = " ".join(contexts).lower()
    q = question.lower()

    # Lightweight extractive behavior for faithfulness
    if "sarvam ai" in q and "sarvam" in blob:
        return "Sarvam AI is an India-first sovereign AI company building multilingual models for Indian languages."
    if "bulbul" in q:
        return "Bulbul is Sarvam's text-to-speech family using latent space decomposition for content vs speaker/style."
    if "saaras" in q:
        return "Saaras is Sarvam's speech-to-text offering targeting low-latency Indic transcription."
    if "800" in blob or ("latency" in q and ("stt" in q or "interactive" in q or "speech" in q)):
        return "Interactive STT systems often target p95 latency under 800ms."
    if "recall" in q or "mrr" in q or "ndcg" in q:
        return "Retrieval quality is measured with Recall@k and MRR, plus nDCG."
    if "rerank" in q or "cross-encoder" in q or "first-stage" in q:
        return "A reranker reorders first-stage candidates to improve precision at small k."
    if "don't know" in q or "missing evidence" in q or "faithful" in q or "insufficient" in q:
        return "A faithful RAG system should say I don't know when evidence is missing."
    if ("cost" in q or "latency" in q) and (
        "production" in q or "report" in q or "kpi" in q or "1000" in q
    ):
        return "Report p50/p95 latency and estimated cost per 1000 queries."
    if "classical" in q or "gradient" in q or "roc" in q or "bias-variance" in q or "bias–variance" in q:
        return "Interview pivots often cover gradient descent, regularization, bias-variance, ROC-AUC, and tree ensembles."
    if "cosine" in q or "embedding" in q or "similarity" in q:
        return "Dense embeddings are typically scored with cosine similarity."
    if "per language" in q or "indian nlp" in q or "code-mixing" in q or "indic" in q:
        return "Evaluate Indian NLP per language due to code-mixing, dialects, and script diversity."
    if "hallucin" in q or "adversarial" in q or "unanswerable" in q:
        return "Adversarial evaluation uses unanswerable questions; a faithful system should abstain."
    if "faithfulness" in q:
        return "Faithfulness measures whether answer claims are supported by retrieved context."

    # Provider-flavored stubs (differentiated offline scores)
    prefix = {
        "openai": "",
        "anthropic": "Carefully: ",
        "google": "",
        "gemma": "",
        "qwen": "Analysis: ",
        "deepseek": "Step-by-step: ",
        "sarvam": "Indic context: ",
    }.get(provider, "")

    sent = re.split(r"[.!?]", contexts[0])[0].strip()
    if style == "verbose":
        return f"{prefix}Based on the context: {sent}."
    if style in {"analytical", "indic"}:
        return f"{prefix}{sent}." if sent else "I don't know"
    return (prefix + sent).strip() or "I don't know"


def _http_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float = 60.0,
) -> dict:
    """POST JSON via stdlib urllib (optional httpx if importable)."""
    body = json.dumps(payload).encode("utf-8")
    try:
        import httpx  # optional

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, content=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        pass

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _live_openai_compatible(
    api_key: str,
    model: str,
    prompt: str,
    base_url: str = "https://api.openai.com/v1",
) -> tuple[str, int, int]:
    data = _http_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided context. "
                        "If evidence is missing, reply exactly: I don't know"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 256,
        },
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    choice = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    return (
        choice,
        int(usage.get("prompt_tokens") or _estimate_tokens(prompt)),
        int(usage.get("completion_tokens") or _estimate_tokens(choice)),
    )


def _live_anthropic(api_key: str, model: str, prompt: str) -> tuple[str, int, int]:
    data = _http_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 256,
            "temperature": 0,
            "system": (
                "Answer using only the provided context. "
                "If evidence is missing, reply exactly: I don't know"
            ),
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    parts = data.get("content") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    usage = data.get("usage") or {}
    return (
        text,
        int(usage.get("input_tokens") or _estimate_tokens(prompt)),
        int(usage.get("output_tokens") or _estimate_tokens(text)),
    )


def _live_google(api_key: str, model: str, prompt: str) -> tuple[str, int, int]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    data = _http_json(
        url,
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
        },
        {"Content-Type": "application/json"},
    )
    cands = data.get("candidates") or []
    text = ""
    if cands:
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
    return text or "I don't know", _estimate_tokens(prompt), _estimate_tokens(text)


def _live_sarvam(api_key: str, model: str, prompt: str) -> tuple[str, int, int]:
    # Sarvam OpenAI-compatible chat endpoint
    return _live_openai_compatible(
        api_key,
        model,
        prompt,
        base_url=os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"),
    )


def live_generate(
    provider: str,
    model: str,
    prompt: str,
) -> tuple[str, int, int] | None:
    """Call a real provider when key is present. Return None to fall back to stub."""
    key = provider_api_key(provider)
    if not key:
        return None

    if provider == "openai":
        return _live_openai_compatible(key, model, prompt)
    if provider == "anthropic":
        return _live_anthropic(key, model, prompt)
    if provider in {"google", "gemma"}:
        return _live_google(key, model, prompt)
    if provider == "sarvam":
        return _live_sarvam(key, model, prompt)
    if provider == "deepseek":
        return _live_openai_compatible(
            key,
            model,
            prompt,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
    if provider == "qwen":
        # Optional DashScope / OpenAI-compatible; reuse OpenAI key + base if set
        base = os.environ.get("QWEN_BASE_URL", "https://api.openai.com/v1")
        return _live_openai_compatible(key, model, prompt, base_url=base)
    return None


def generate_answer(
    provider: str,
    model: str,
    prompt: str,
    question: str,
    contexts: list[str],
    style: str = "concise",
    unanswerable: bool = False,
) -> GenerationResult:
    """Prefer stub unless EVAL_LIVE=1 and a provider API key is present."""
    import time

    t0 = time.perf_counter()
    mode = "stub"
    answer: str
    prompt_tokens: int
    completion_tokens: int

    if live_enabled():
        try:
            live = live_generate(provider, model, prompt)
        except Exception as exc:  # noqa: BLE001 — fall back so lab never hard-crashes
            print(f"[warn] live {provider} failed ({exc}); using stub")
            live = None
        if live is not None:
            answer, prompt_tokens, completion_tokens = live
            mode = "live"
        else:
            answer = stub_answer(
                provider,
                question,
                contexts,
                style=style,
                force_abstain=unanswerable,
            )
            prompt_tokens = _estimate_tokens(prompt)
            completion_tokens = _estimate_tokens(answer)
    else:
        answer = stub_answer(
            provider,
            question,
            contexts,
            style=style,
            force_abstain=unanswerable,
        )
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(answer)

    latency_bias = {
        "openai": 12,
        "anthropic": 18,
        "google": 10,
        "qwen": 15,
        "gemma": 8,
        "deepseek": 11,
        "sarvam": 14,
    }.get(provider, 12)
    elapsed = (time.perf_counter() - t0) * 1000
    if mode == "stub":
        elapsed += latency_bias

    return GenerationResult(
        provider=provider,
        model=model,
        answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=elapsed,
        mode=mode,
    )
