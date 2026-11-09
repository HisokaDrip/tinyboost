import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_friedman1
from sklearn.model_selection import train_test_split

from tinyboost import BinMapper, GBClassifier, GBRegressor
from tinyboost.tree import TreeParams, build_histogram, find_best_split, grow_tree


def test_binning_is_monotone_and_reserves_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    X[::17, 1] = np.nan
    bm = BinMapper(max_bins=32).fit(X)
    Xb = bm.transform(X)
    assert Xb[~np.isnan(X[:, 0]), 0].max() < 32
    assert np.all(Xb[np.isnan(X[:, 1]), 1] == bm.missing_bin)
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(Xb[order, 0].astype(int)) >= 0)


def test_low_cardinality_feature_gets_exact_bins():
    X = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
    bm = BinMapper().fit(X)
    assert list(bm.transform(X)[:, 0]) == [0, 1, 2, 1, 0]


def test_histogram_matches_brute_force():
    rng = np.random.default_rng(1)
    Xb = rng.integers(0, 8, size=(200, 4)).astype(np.uint8)
    g, h = rng.normal(size=200), rng.random(200)
    rows = np.arange(0, 200, 3)
    hist = build_histogram(Xb, rows, g, h, 8)
    for f in range(4):
        for b in range(8):
            sel = rows[Xb[rows, f] == b]
            assert hist[f, b, 0] == pytest.approx(g[sel].sum())
            assert hist[f, b, 1] == pytest.approx(h[sel].sum())
            assert hist[f, b, 2] == len(sel)


def test_split_finds_the_obvious_cut():
    # feature 2 is a perfect step function, the others are noise
    rng = np.random.default_rng(2)
    Xb = rng.integers(0, 10, size=(500, 3)).astype(np.uint8)
    g = np.where(Xb[:, 2] <= 4, -1.0, 1.0)
    h = np.ones(500)
    hist = build_histogram(Xb, np.arange(500), g, h, 11)
    split = find_best_split(hist, TreeParams(min_samples_leaf=1))
    assert (split.feature, split.bin) == (2, 4)


def test_nan_direction_is_learned():
    # NaN rows behave like the high side, so they should be routed right
    rng = np.random.default_rng(3)
    x = rng.random(2000)
    y = (x > 0.5).astype(float)
    x[rng.random(2000) < 0.2] = np.nan
    y[np.isnan(x)] = 1.0
    m = GBRegressor(n_estimators=30, max_leaves=2, learning_rate=0.5).fit(x[:, None], y)
    pred = m.predict(np.array([[np.nan], [0.1], [0.9]]))
    assert pred[0] > 0.9 and pred[1] < 0.1 and pred[2] > 0.9


def test_histogram_subtraction_matches_direct_build():
    rng = np.random.default_rng(4)
    Xb = rng.integers(0, 16, size=(300, 5)).astype(np.uint8)
    g, h = rng.normal(size=300), rng.random(300) + 0.1
    rows = np.arange(300)
    left, right = rows[Xb[:, 0] <= 7], rows[Xb[:, 0] > 7]
    parent = build_histogram(Xb, rows, g, h, 16)
    direct = build_histogram(Xb, right, g, h, 16)
    np.testing.assert_allclose(parent - build_histogram(Xb, left, g, h, 16), direct, atol=1e-10)


def test_max_leaves_is_respected():
    X, y = make_friedman1(2000, random_state=0)
    m = GBRegressor(n_estimators=5, max_leaves=7).fit(X, y)
    assert all(t.n_leaves <= 7 for t in m.trees_)
    m = GBRegressor(n_estimators=5, max_leaves=64, max_depth=2).fit(X, y)
    assert all(t.n_leaves <= 4 for t in m.trees_)


def test_training_loss_goes_down():
    X, y = make_friedman1(1500, random_state=1)
    m = GBRegressor(n_estimators=50).fit(X, y)
    losses = [r["train"] for r in m.history_]
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))


def test_regressor_beats_mean_by_a_lot():
    X, y = make_friedman1(4000, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    m = GBRegressor(n_estimators=300, early_stopping_rounds=20).fit(Xtr, ytr)
    r2 = 1 - np.mean((m.predict(Xte) - yte) ** 2) / np.var(yte)
    assert r2 > 0.9
    assert m.best_iteration_ < 300 or len(m.trees_) == 300


def test_classifier_on_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
    m = GBClassifier(n_estimators=150, subsample=0.8, colsample=0.7).fit(Xtr, ytr)
    assert (m.predict(Xte) == yte).mean() > 0.94
    proba = m.predict_proba(Xte)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_keeps_original_labels():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 2))
    y = np.where(X[:, 0] > 0, "spam", "ham")
    m = GBClassifier(n_estimators=20).fit(X, y)
    assert set(m.predict(X)) <= {"spam", "ham"}
    assert (m.predict(X) == y).mean() > 0.95


def test_importances_point_at_informative_features():
    # friedman1: only x0..x4 matter, x5..x9 are pure noise
    X, y = make_friedman1(3000, n_features=10, random_state=0)
    imp = GBRegressor(n_estimators=100).fit(X, y).feature_importances_
    assert imp.sum() == pytest.approx(1.0)
    assert imp[:5].sum() > 0.95


def test_multiclass_is_rejected():
    with pytest.raises(ValueError):
        GBClassifier().fit(np.zeros((6, 1)), [0, 1, 2, 0, 1, 2])


def test_grow_tree_single_leaf_when_nothing_to_gain():
    Xb = np.zeros((50, 2), dtype=np.uint8)
    t = grow_tree(Xb, np.ones(50), np.ones(50), np.arange(50), 4, TreeParams())
    assert t.n_leaves == 1
    assert t.value[0] == pytest.approx(-50 / 51)
