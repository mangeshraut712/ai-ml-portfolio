from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.answer_quality import token_f1


def test_token_f1_counts_duplicate_tokens_as_multiset() -> None:
    # Set-based overlap would give F1=1.0 here (both reduce to {"bandwidth"});
    # the metric must charge precision for repeated tokens instead.
    pred = "bandwidth bandwidth bandwidth"
    gold = "bandwidth"
    # common=1, precision=1/3, recall=1 -> F1 = 0.5
    assert token_f1(pred, gold) == pytest.approx(0.5)
    assert token_f1(gold, pred) == pytest.approx(0.5)


def test_token_f1_is_case_and_order_insensitive() -> None:
    assert token_f1("The cat, sat!", "sat the cat") == pytest.approx(1.0)


def test_token_f1_tokenizer_keeps_apostrophes() -> None:
    assert token_f1("I don't know.", "i don't know") == pytest.approx(1.0)
    # "dont" and "don't" are distinct tokens: 2 of 3 overlap -> F1 = 2/3
    assert token_f1("I dont know", "I don't know") == pytest.approx(2 / 3)


def test_token_f1_empty_string_conventions() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("answer", "") == 0.0
    assert token_f1("", "answer") == 0.0
