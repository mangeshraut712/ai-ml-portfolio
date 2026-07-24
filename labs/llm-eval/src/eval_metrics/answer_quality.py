"""Answer overlap helpers for prompt evaluation."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9']+")


def token_f1(pred: str, gold: str) -> float:
    p = _TOKEN.findall(pred.lower())
    g = _TOKEN.findall(gold.lower())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    bag = list(g)
    for t in p:
        if t in bag:
            bag.remove(t)
            common += 1
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
