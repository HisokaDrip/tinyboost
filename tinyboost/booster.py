import time

import numpy as np

from .binning import BinMapper
from .losses import LogLoss, SquaredError
from .tree import TreeParams, grow_tree


class _Booster:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_leaves=31, max_depth=-1,
                 min_samples_leaf=20, min_child_weight=1e-3, reg_lambda=1.0, gamma=0.0,
                 subsample=1.0, colsample=1.0, max_bins=255, early_stopping_rounds=None,
                 validation_fraction=0.1, random_state=0, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_params = TreeParams(max_leaves, max_depth, min_child_weight,
                                      min_samples_leaf, reg_lambda, gamma)
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, eval_set=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        if self.early_stopping_rounds and eval_set is None:
            perm = rng.permutation(len(y))
            n_val = max(1, int(len(y) * self.validation_fraction))
            val, tr = perm[:n_val], perm[n_val:]
            eval_set = (X[val], y[val])
            X, y = X[tr], y[tr]

        self.bins_ = BinMapper(self.max_bins, seed=self.random_state)
        Xb = self.bins_.fit_transform(X)
        n_bins = self.bins_.n_bins
        self.n_features_ = X.shape[1]

        self.base_score_ = self.loss_.init_score(y)
        raw = np.full(len(y), self.base_score_)
        if eval_set is not None:
            Xv_b = self.bins_.transform(eval_set[0])
            yv = np.asarray(eval_set[1], dtype=np.float64)
            raw_v = np.full(len(yv), self.base_score_)

        self.trees_, self.history_ = [], []
        best, best_iter, t0 = np.inf, 0, time.perf_counter()
        all_rows = np.arange(len(y))

        for it in range(self.n_estimators):
            g, h = self.loss_.grad_hess(y, raw)
            rows = all_rows
            if self.subsample < 1.0:
                rows = np.sort(rng.choice(len(y), int(len(y) * self.subsample), replace=False))
            fmask = None
            if self.colsample < 1.0:
                k = max(1, int(round(self.n_features_ * self.colsample)))
                fmask = np.zeros(self.n_features_, dtype=bool)
                fmask[rng.choice(self.n_features_, k, replace=False)] = True

            tree = grow_tree(Xb, g, h, rows, n_bins, self.tree_params, fmask,
                             self.learning_rate)
            self.trees_.append(tree)
            raw += tree.predict_binned(Xb, self.bins_.missing_bin)

            rec = {"iter": it + 1, "train": self.loss_.loss(y, raw)}
            if eval_set is not None:
                raw_v += tree.predict_binned(Xv_b, self.bins_.missing_bin)
                rec["valid"] = self.loss_.loss(yv, raw_v)
                if rec["valid"] < best - 1e-12:
                    best, best_iter = rec["valid"], it + 1
            self.history_.append(rec)

            if self.verbose and (it + 1) % self.verbose == 0:
                extra = f"  valid {rec['valid']:.5f}" if "valid" in rec else ""
                print(f"[{it + 1:4d}] train {rec['train']:.5f}{extra}"
                      f"  ({time.perf_counter() - t0:.1f}s)")

            if (self.early_stopping_rounds and eval_set is not None
                    and it + 1 - best_iter >= self.early_stopping_rounds):
                break

        if self.early_stopping_rounds and eval_set is not None:
            self.trees_ = self.trees_[:best_iter]
        self.best_iteration_ = len(self.trees_)
        return self

    def raw_predict(self, X):
        Xb = self.bins_.transform(X)
        out = np.full(Xb.shape[0], self.base_score_)
        for tree in self.trees_:
            out += tree.predict_binned(Xb, self.bins_.missing_bin)
        return out

    @property
    def feature_importances_(self):
        # total split gain per feature, normalised. gain beats split counts
        # because it doesn't reward features that get chopped into tiny bits
        imp = np.zeros(self.n_features_)
        for t in self.trees_:
            mask = t.feature >= 0
            np.add.at(imp, t.feature[mask], t.gain[mask])
        s = imp.sum()
        return imp / s if s > 0 else imp

    def dump_tree(self, i=0, feature_names=None):
        t = self.trees_[i]
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def walk(n, depth):
            pad = "  " * depth
            if t.feature[n] < 0:
                lines.append(f"{pad}leaf {t.value[n]:+.4f}")
                return
            thr = self.bins_.threshold(t.feature[n], t.bin[n])
            nan = "  (nan -> left)" if t.missing_left[n] else ""
            lines.append(f"{pad}{names[t.feature[n]]} <= {thr:.4g}{nan}")
            walk(t.left[n], depth + 1)
            walk(t.right[n], depth + 1)

        walk(0, 0)
        return "\n".join(lines)


class GBRegressor(_Booster):
    loss_ = SquaredError()

    def predict(self, X):
        return self.raw_predict(X)


class GBClassifier(_Booster):
    """Binary only for now."""
    loss_ = LogLoss()

    def fit(self, X, y, eval_set=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GBClassifier only does binary classification")
        yy = (y == self.classes_[1]).astype(np.float64)
        if eval_set is not None:
            eval_set = (eval_set[0], (np.asarray(eval_set[1]) == self.classes_[1]).astype(float))
        return super().fit(X, yy, eval_set)

    def predict_proba(self, X):
        p = self.loss_.transform(self.raw_predict(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.raw_predict(X) > 0).astype(int)]
