import heapq
from dataclasses import dataclass

import numpy as np

LEAF = -1


@dataclass
class TreeParams:
    max_leaves: int = 31
    max_depth: int = -1          # -1 = no limit, leaf count is the real cap
    min_child_weight: float = 1e-3   # minimum hessian sum in a child
    min_samples_leaf: int = 20
    reg_lambda: float = 1.0
    gamma: float = 0.0           # minimum gain to bother splitting


def build_histogram(Xb, rows, g, h, n_bins):
    """(n_features, n_bins, 3) array of grad sum, hess sum, count per bin.

    One bincount over the whole node instead of a python loop per feature:
    shift each feature's codes into its own block of n_bins slots.
    """
    n_feat = Xb.shape[1]
    codes = Xb[rows].astype(np.intp) + np.arange(n_feat) * n_bins
    codes = codes.ravel()
    size = n_feat * n_bins
    gr = np.repeat(g[rows], n_feat)
    hr = np.repeat(h[rows], n_feat)
    hist = np.empty((size, 3))
    hist[:, 0] = np.bincount(codes, weights=gr, minlength=size)
    hist[:, 1] = np.bincount(codes, weights=hr, minlength=size)
    hist[:, 2] = np.bincount(codes, minlength=size)
    return hist.reshape(n_feat, n_bins, 3)


def leaf_weight(G, H, lam):
    return -G / (H + lam)


def _score(G, H, lam):
    return G * G / (H + lam)


@dataclass
class Split:
    gain: float
    feature: int
    bin: int
    missing_left: bool


def find_best_split(hist, p: TreeParams, feature_mask=None):
    """Scan every (feature, bin) cut, trying NaNs on both sides.

    hist[:, :-1] are the ordered real bins, hist[:, -1] is the NaN bin.
    """
    real = hist[:, :-1, :]
    miss = hist[:, -1, :]                       # (n_feat, 3)
    cum = np.cumsum(real, axis=1)[:, :-1, :]    # left side if we cut after bin b
    total = real.sum(axis=1) + miss             # same for every feature
    G, H = total[0, 0], total[0, 1]
    lam = p.reg_lambda
    parent = _score(G, H, lam)

    best = None
    for missing_left in (False, True):
        left = cum + miss[:, None, :] if missing_left else cum
        GL, HL, NL = left[..., 0], left[..., 1], left[..., 2]
        GR, HR, NR = G - GL, H - HL, total[:, None, 2] - NL
        gain = 0.5 * (_score(GL, HL, lam) + _score(GR, HR, lam) - parent) - p.gamma
        bad = ((HL < p.min_child_weight) | (HR < p.min_child_weight)
               | (NL < p.min_samples_leaf) | (NR < p.min_samples_leaf))
        if feature_mask is not None:
            bad |= ~feature_mask[:, None]
        gain = np.where(bad, -np.inf, gain)
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[f, b] > 0 and (best is None or gain[f, b] > best.gain):
            best = Split(float(gain[f, b]), int(f), int(b), missing_left)
    return best


class Tree:
    """Flat-array tree. Children are indices into the same arrays."""

    def __init__(self):
        self.feature, self.bin, self.missing_left = [], [], []
        self.left, self.right, self.value, self.gain = [], [], [], []

    def _add(self, value):
        for arr, v in ((self.feature, LEAF), (self.bin, 0), (self.missing_left, False),
                       (self.left, LEAF), (self.right, LEAF), (self.value, value),
                       (self.gain, 0.0)):
            arr.append(v)
        return len(self.value) - 1

    def finalize(self):
        for name in ("feature", "bin", "left", "right"):
            setattr(self, name, np.array(getattr(self, name), dtype=np.int32))
        self.missing_left = np.array(self.missing_left, dtype=bool)
        self.value = np.array(self.value, dtype=np.float64)
        self.gain = np.array(self.gain, dtype=np.float64)
        return self

    @property
    def n_leaves(self):
        return int(np.sum(self.feature == LEAF))

    def predict_binned(self, Xb, missing_bin):
        node = np.zeros(Xb.shape[0], dtype=np.int32)
        active = self.feature[node] != LEAF
        # walk all rows down one level at a time; depth is small so this is cheap
        while active.any():
            idx = np.nonzero(active)[0]
            nd = node[idx]
            codes = Xb[idx, self.feature[nd]]
            go_left = np.where(codes == missing_bin, self.missing_left[nd], codes <= self.bin[nd])
            node[idx] = np.where(go_left, self.left[nd], self.right[nd])
            active[idx] = self.feature[node[idx]] != LEAF
        return self.value[node]


def grow_tree(Xb, g, h, rows, n_bins, p: TreeParams, feature_mask=None, learning_rate=1.0):
    """Leaf-wise (best-first) growth, LightGBM style.

    Always split the leaf with the biggest gain next. Histograms for the
    bigger child come from parent - smaller child, which roughly halves the
    work per level.
    """
    tree = Tree()
    lam = p.reg_lambda

    def make_leaf(rows, hist, depth):
        G, H = hist[0, :, 0].sum(), hist[0, :, 1].sum()
        nid = tree._add(learning_rate * leaf_weight(G, H, lam))
        split = None
        if p.max_depth < 0 or depth < p.max_depth:
            split = find_best_split(hist, p, feature_mask)
        return nid, rows, hist, depth, split

    heap, counter = [], 0

    def push(leaf):
        nonlocal counter
        if leaf[4] is not None:
            heapq.heappush(heap, (-leaf[4].gain, counter, leaf))
            counter += 1

    push(make_leaf(rows, build_histogram(Xb, rows, g, h, n_bins), 0))
    n_leaves = 1
    while heap and n_leaves < p.max_leaves:
        _, _, (nid, rows, hist, depth, split) = heapq.heappop(heap)
        codes = Xb[rows, split.feature]
        if split.missing_left:
            go_left = (codes <= split.bin) | (codes == n_bins - 1)
        else:
            go_left = codes <= split.bin
        lrows, rrows = rows[go_left], rows[~go_left]

        if len(lrows) <= len(rrows):
            lhist = build_histogram(Xb, lrows, g, h, n_bins)
            rhist = hist - lhist
        else:
            rhist = build_histogram(Xb, rrows, g, h, n_bins)
            lhist = hist - rhist

        left = make_leaf(lrows, lhist, depth + 1)
        right = make_leaf(rrows, rhist, depth + 1)
        tree.feature[nid] = split.feature
        tree.bin[nid] = split.bin
        tree.missing_left[nid] = split.missing_left
        tree.left[nid], tree.right[nid] = left[0], right[0]
        tree.gain[nid] = split.gain
        push(left)
        push(right)
        n_leaves += 1

    return tree.finalize()
