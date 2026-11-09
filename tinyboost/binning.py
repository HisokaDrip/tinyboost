import numpy as np


class BinMapper:
    """Quantile-bins every feature into at most `max_bins` uint8 codes.

    The last code (max_bins) is reserved for NaN, so a node can learn which
    way missing values should go instead of us imputing something dumb.
    """

    def __init__(self, max_bins=255, subsample=200_000, seed=0):
        if not 2 <= max_bins <= 255:
            raise ValueError("max_bins must be in [2, 255]")
        self.max_bins = max_bins
        self.subsample = subsample
        self.seed = seed

    @property
    def missing_bin(self):
        return self.max_bins

    @property
    def n_bins(self):
        return self.max_bins + 1  # real bins + the NaN bin

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > self.subsample:
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(X.shape[0], self.subsample, replace=False)]

        self.edges_ = []
        for col in X.T:
            col = col[~np.isnan(col)]
            uniq = np.unique(col)
            if uniq.size <= self.max_bins:
                # few distinct values -> split exactly between neighbours
                edges = (uniq[:-1] + uniq[1:]) / 2
            else:
                qs = np.linspace(0, 100, self.max_bins + 1)[1:-1]
                edges = np.unique(np.percentile(col, qs, method="midpoint"))
            self.edges_.append(edges)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.edges_):
            col = X[:, j]
            codes = np.searchsorted(edges, col, side="right")
            codes[np.isnan(col)] = self.missing_bin
            out[:, j] = codes
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def threshold(self, feature, bin_idx):
        # real-valued threshold for "x <= t goes left", handy for dumping trees
        edges = self.edges_[feature]
        return float(edges[bin_idx]) if bin_idx < len(edges) else float("inf")
