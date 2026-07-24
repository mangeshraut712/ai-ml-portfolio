"""Multilayer perceptron with manual backprop (NumPy)."""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    """One-hidden-layer MLP for multiclass / binary (2-class) problems."""

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 0.05,
        n_iter: int = 800,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1).astype(int)
        n, d = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        k = len(classes)
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        Y = np.zeros((n, k))
        for i, yi in enumerate(y):
            Y[i, class_to_idx[int(yi)]] = 1.0

        self.W1_ = rng.normal(0, 0.1, size=(d, self.hidden))
        self.b1_ = np.zeros(self.hidden)
        self.W2_ = rng.normal(0, 0.1, size=(self.hidden, k))
        self.b2_ = np.zeros(k)

        for _ in range(self.n_iter):
            # forward
            z1 = X @ self.W1_ + self.b1_
            a1 = relu(z1)
            z2 = a1 @ self.W2_ + self.b2_
            p = softmax(z2)
            # backprop
            dz2 = (p - Y) / n
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.W2_.T
            dz1 = da1 * relu_grad(z1)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            self.W2_ -= self.lr * dW2
            self.b2_ -= self.lr * db2
            self.W1_ -= self.lr * dW1
            self.b1_ -= self.lr * db1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        a1 = relu(X @ self.W1_ + self.b1_)
        return softmax(a1 @ self.W2_ + self.b2_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]
