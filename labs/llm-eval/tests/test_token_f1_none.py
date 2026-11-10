import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_metrics.answer_quality import token_f1


def test_token_f1_rejects_none() -> None:
    with pytest.raises(TypeError, match="not None"):
        token_f1(None, "hello")
    with pytest.raises(TypeError, match="not None"):
        token_f1("hello", None)
    assert token_f1("hello world", "hello there") > 0
