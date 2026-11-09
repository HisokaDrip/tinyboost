# tinyboost

A histogram gradient boosting library in about 450 lines of NumPy. It's the same
family as LightGBM / XGBoost `hist` / sklearn's `HistGradientBoosting`, written
so you can actually read how it works.

I've used gradient boosting on basically every tabular problem I've touched and
wanted to know what's really going on under `.fit()`. Turns out the core idea is
small, and most of the speed tricks are clever bookkeeping rather than magic.

## What's in it

- **Quantile binning** – every feature is bucketed into at most 255 `uint8` codes
  up front, so finding a split is a scan over 255 bins instead of sorting rows.
  Low-cardinality features get exact bins.
- **Second-order (Newton) boosting** – each tree fits gradients *and* hessians,
  leaf weight is `-G / (H + lambda)`, split gain is the usual XGBoost formula.
- **Leaf-wise growth** – always split the leaf with the best gain next (LightGBM
  style), capped by `max_leaves` and optionally `max_depth`.
- **Histogram subtraction** – only build the histogram for the smaller child;
  the bigger one is `parent - smaller`. The whole node histogram is a single
  `np.bincount` across all features.
- **Missing values** – NaNs get their own bin, and every split tries sending them
  left and right and keeps whichever gains more. No imputation needed.
- Squared error and log loss, row subsampling, column subsampling, L2 + `gamma`,
  early stopping on a held-out set, gain-based feature importances, and a tree
  dumper that prints real thresholds.

## Usage

```python
from tinyboost import GBClassifier, GBRegressor

clf = GBClassifier(n_estimators=300, learning_rate=0.1, max_leaves=31,
                   subsample=0.8, colsample=0.8, early_stopping_rounds=20)
clf.fit(X_train, y_train)          # NaNs in X are fine
clf.predict_proba(X_test)
clf.feature_importances_
print(clf.dump_tree(0))
```

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bench.py
pytest -q
```

## Results

`bench.py` runs both libraries with the same settings (lr 0.1, 31 leaves,
min 20 samples per leaf, l2 = 1, 255 bins, no early stopping) on the same
splits. Output from my run:

```
classification
  breast_cancer (569x30)     tinyboost  acc 0.9510  auc 0.9853  fit  1.28s
  breast_cancer (569x30)     sklearn    acc 0.9441  auc 0.9870  fit  0.15s
  hastie_10_2 (12000x10)     tinyboost  acc 0.9433  auc 0.9909  fit  4.79s
  hastie_10_2 (12000x10)     sklearn    acc 0.9467  auc 0.9916  fit  0.60s
  hastie + 15% NaNs          tinyboost  acc 0.8647  auc 0.9371  fit  4.71s
  hastie + 15% NaNs          sklearn    acc 0.8590  auc 0.9372  fit  1.02s

regression
  friedman1 (20000x10)       tinyboost  rmse 1.1064  r2 0.9504  fit  4.66s
  friedman1 (20000x10)       sklearn    rmse 1.1068  r2 0.9504  fit  0.76s
  friedman1 + 15% NaNs       tinyboost  rmse 2.3261  r2 0.7807  fit  4.61s
  friedman1 + 15% NaNs       sklearn    rmse 2.3296  r2 0.7801  fit  0.82s
```

So accuracy is basically identical to sklearn (which is the point: same
algorithm), and it's 6-8x slower, which honestly is better than I expected for
pure NumPy against Cython + OpenMP.

The first tree on friedman1, and the gain importances. Friedman1 only uses
x0-x4, x5-x9 are pure noise, and the model figures that out:

```
x3 <= 0.4556
  x1 <= 0.3444
    leaf -5.4463
    x0 <= 0.3024
      leaf -4.8673
      leaf +0.2828
  x1 <= 0.2975
    leaf -0.7467
    x0 <= 0.3061
      leaf +0.0292
      leaf +5.1412

gain importances: x0=0.235 x1=0.236 x2=0.089 x3=0.348 x4=0.088 x5=0.001 x6=0.001 x7=0.001 x8=0.001 x9=0.001
```

(leaf values are offsets on top of the mean of y.)

## Tests

`pytest -q` checks the histogram against a brute-force count, that subtraction
gives the same histogram as building it directly, that the split finder picks an
obvious step, that NaNs get routed the right way, leaf limits, monotone training
loss, and end-to-end accuracy on real datasets.

## What I'd do next

- multiclass softmax (one tree per class per round)
- categorical splits (sort categories by G/H, then it's just another ordered split)
- speed: a numba version of `build_histogram` and the prediction walk would
  probably close a lot of the gap with sklearn
