"""Unit tests for classical ML from scratch."""

from __future__ import annotations

import numpy as np

from mlfs.attention import scaled_dot_product_attention
from mlfs.calibration import brier_score, expected_calibration_error
from mlfs.gradient_boosting import GradientBoostingBinary
from mlfs.kmeans import KMeans
from mlfs.knn import KNNClassifier
from mlfs.linear_regression import LinearRegression
from mlfs.logistic_regression import LogisticRegression
from mlfs.metrics import accuracy, roc_auc_score
from mlfs.mlp import MLPClassifier
from mlfs.naive_bayes import GaussianNB
from mlfs.pca import PCA
from mlfs.preprocessing import StandardScaler
from mlfs.random_forest import RandomForestClassifier
from mlfs.svm import LinearSVM


def _cls(seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1.2, -1.2], 0.5, size=(40, 2))
    X1 = rng.normal([1.2, 1.2], 0.5, size=(40, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 40 + [1] * 40)
    return StandardScaler().fit_transform(X), y


def test_linear_regression_fits_line():
    X = np.linspace(0, 1, 50)[:, None]
    y = 3 * X[:, 0] + 2
    model = LinearRegression(lr=0.1, n_iter=3000).fit(X, y)
    pred = model.predict(X)
    assert np.mean((pred - y) ** 2) < 1e-3


def test_logistic_and_auc():
    X, y = _cls()
    clf = LogisticRegression(lr=0.5, n_iter=2000, l2=0.01).fit(X, y)
    assert accuracy(y, clf.predict(X)) > 0.9
    assert roc_auc_score(y, clf.predict_proba(X)) > 0.9


def test_knn_nb_svm_forest():
    X, y = _cls()
    assert accuracy(y, KNNClassifier(k=3).fit(X, y).predict(X)) > 0.85
    assert accuracy(y, GaussianNB().fit(X, y).predict(X)) > 0.85
    assert accuracy(y, LinearSVM(n_iter=1500).fit(X, y).predict(X)) > 0.85
    assert (
        accuracy(y, RandomForestClassifier(n_estimators=8, max_depth=4).fit(X, y).predict(X))
        > 0.85
    )


def test_pca_kmeans():
    X, _ = _cls()
    Z = PCA(n_components=1).fit_transform(X)
    assert Z.shape == (80, 1)
    labels = KMeans(n_clusters=2, seed=0).fit(X).predict(X)
    assert len(np.unique(labels)) == 2


def test_mlp_backprop():
    X, y = _cls()
    clf = MLPClassifier(hidden=16, lr=0.1, n_iter=600, seed=0).fit(X, y)
    assert accuracy(y, clf.predict(X)) > 0.9


def test_attention_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 5, 8))
    out, weights = scaled_dot_product_attention(X, X, X)
    assert out.shape == X.shape
    assert weights.shape == (2, 5, 5)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_gbdt_and_calibration():
    X, y = _cls()
    clf = GradientBoostingBinary(n_estimators=15, max_depth=2, learning_rate=0.2).fit(X, y)
    proba = clf.predict_proba(X)
    assert accuracy(y, clf.predict(X)) > 0.85
    assert brier_score(y, proba) < 0.2
    assert expected_calibration_error(y, proba) < 0.35
