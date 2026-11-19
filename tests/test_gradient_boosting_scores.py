from __future__ import annotations

import numpy as np

from mlfs.gradient_boosting import GradientBoostingBinary


def test_decision_function_uses_the_step_sizes_from_fit() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    model = GradientBoostingBinary(n_estimators=1, learning_rate=0.5).fit(X, y)

    first_residual = y - 1 / (1 + np.exp(-model.init_))
    expected_step = model.learning_rate * np.abs(first_residual).mean()
    steps = np.abs(model.decision_function(X) - model.init_)

    np.testing.assert_allclose(steps, expected_step)
    assert expected_step < model.learning_rate  # differs from the old unscaled step
