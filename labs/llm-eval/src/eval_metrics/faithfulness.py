"""Faithfulness and hallucination heuristics (offline-friendly)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """Lexical support: fraction of answer content tokens found in context.

    Abstentions ('i don't know') score 1.0 when intentional.
    """
    ans = answer.strip().lower()
    if ans in {"i don't know", "i do not know", "unknown", "n/a"}:
        return 1.0
    a_toks = _tokens(answer)
    if not a_toks:
        return 0.0
    ctx = _tokens(" ".join(contexts))
    # Drop ultra-common function words for a stabler signal
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "on"}
    a_toks -= stop
    if not a_toks:
        return 1.0
    hit = sum(1 for t in a_toks if t in ctx)
    return hit / len(a_toks)


def is_abstention(answer: str) -> bool:
    ans = answer.strip().lower()
    return ans.startswith("i don't know") or ans.startswith("i do not know")


def hallucination_on_unanswerable(answer: str) -> bool:
    """True if model fabricated an answer instead of abstaining."""
    return not is_abstention(answer)
