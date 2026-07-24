"""Run printable demos for interviews."""

from __future__ import annotations

import numpy as np

from mlfs.bias_variance import bias_variance_curve
from mlfs.logistic_regression import LogisticRegression
from mlfs.metrics import accuracy, roc_auc_score
from mlfs.preprocessing import StandardScaler


def _blob(seed: int = 0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1, -1], 0.6, size=(60, 2))
    X1 = rng.normal([1, 1], 0.6, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


def run_all() -> None:
    print("=== Bias–Variance (poly degree vs MSE) ===")
    for row in bias_variance_curve():
        print(
            f"  degree={row['degree']:>2}  train={row['train_mse']:.4f}  "
            f"test={row['test_mse']:.4f}"
        )

    print("\n=== Logistic + ROC-AUC ===")
    X, y = _blob()
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(lr=0.5, n_iter=1500, l2=0.01).fit(Xs, y)
    pred = clf.predict(Xs)
    scores = clf.predict_proba(Xs)
    print(f"  accuracy={accuracy(y, pred):.3f}  roc_auc={roc_auc_score(y, scores):.3f}")

    from mlfs.attention import scaled_dot_product_attention
    from mlfs.mlp import MLPClassifier

    print("\n=== MLP (manual backprop) ===")
    mlp = MLPClassifier(hidden=24, lr=0.1, n_iter=500, seed=0).fit(Xs, y)
    print(f"  accuracy={accuracy(y, mlp.predict(Xs)):.3f}")

    print("\n=== Attention sanity ===")
    import numpy as np

    tok = np.random.default_rng(0).normal(size=(1, 4, 8))
    out, w = scaled_dot_product_attention(tok, tok, tok)
    print(f"  out={out.shape}  weights_sum={w.sum(axis=-1).mean():.3f}")


if __name__ == "__main__":
    run_all()
