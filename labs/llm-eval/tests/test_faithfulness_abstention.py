from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.faithfulness import faithfulness_score, is_abstention


def test_faithfulness_agrees_with_is_abstention_on_punctuated_dont_know() -> None:
    for answer in ("I don't know", "I don't know.", "I do not know!", "unknown", "n/a"):
        assert is_abstention(answer)
        assert faithfulness_score(answer, []) == 1.0
    assert not is_abstention("Paris is the capital.")
    assert faithfulness_score("Paris is the capital.", []) == 0.0
