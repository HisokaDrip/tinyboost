"""Head-to-head with sklearn's HistGradientBoosting on the same splits.

Same knobs on both sides (lr 0.1, 31 leaves, 20 min samples, l2 = 1, 255 bins)
so the numbers are actually comparable. sklearn is Cython + OpenMP, we're
numpy, so the interesting part is accuracy, not speed.
"""
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, make_friedman1, make_hastie_10_2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tinyboost import GBClassifier, GBRegressor

COMMON = dict(learning_rate=0.1, min_samples_leaf=20)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def with_missing(X, frac, seed):
    X = X.copy()
    X[np.random.default_rng(seed).random(X.shape) < frac] = np.nan
    return X


def run_clf(name, X, y, n_iter):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ours, t1 = timed(lambda: GBClassifier(n_estimators=n_iter, max_leaves=31, reg_lambda=1.0,
                                          **COMMON).fit(Xtr, ytr))
    skl, t2 = timed(lambda: HistGradientBoostingClassifier(
        max_iter=n_iter, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False,
        random_state=0, **COMMON).fit(Xtr, ytr))
    for who, m, t in (("tinyboost", ours, t1), ("sklearn", skl, t2)):
        p = m.predict_proba(Xte)[:, 1]
        print(f"  {name:<26} {who:<10} acc {accuracy_score(yte, m.predict(Xte)):.4f}"
              f"  auc {roc_auc_score(yte, p):.4f}  fit {t:5.2f}s")


def run_reg(name, X, y, n_iter):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    ours, t1 = timed(lambda: GBRegressor(n_estimators=n_iter, max_leaves=31, reg_lambda=1.0,
                                         **COMMON).fit(Xtr, ytr))
    skl, t2 = timed(lambda: HistGradientBoostingRegressor(
        max_iter=n_iter, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False,
        random_state=0, **COMMON).fit(Xtr, ytr))
    for who, m, t in (("tinyboost", ours, t1), ("sklearn", skl, t2)):
        err = m.predict(Xte) - yte
        r2 = 1 - np.mean(err ** 2) / np.var(yte)
        print(f"  {name:<26} {who:<10} rmse {np.sqrt(np.mean(err ** 2)):.4f}"
              f"  r2 {r2:.4f}  fit {t:5.2f}s")


if __name__ == "__main__":
    print("classification")
    X, y = load_breast_cancer(return_X_y=True)
    run_clf("breast_cancer (569x30)", X, y, 200)
    X, y = make_hastie_10_2(12000, random_state=0)
    run_clf("hastie_10_2 (12000x10)", X, y, 300)
    run_clf("hastie + 15% NaNs", with_missing(X, 0.15, 1), y, 300)

    print("\nregression")
    X, y = make_friedman1(20000, noise=1.0, random_state=0)
    run_reg("friedman1 (20000x10)", X, y, 300)
    run_reg("friedman1 + 15% NaNs", with_missing(X, 0.15, 2), y, 300)

    print("\nwhat the first tree learned on friedman1 (6 leaves so it fits on screen):")
    m = GBRegressor(n_estimators=1, max_leaves=6, learning_rate=1.0).fit(X, y)
    print(m.dump_tree(0))
    imp = GBRegressor(n_estimators=200).fit(X, y).feature_importances_
    print("\ngain importances:", " ".join(f"x{i}={v:.3f}" for i, v in enumerate(imp)))
