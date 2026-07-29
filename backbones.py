"""
Alternative upstream predictors, used for the cross-backbone robustness analysis
(Task 1) and for the septic-shock task (Task 2). Both return out-of-fold
predictions under stay-level cross-fitting.

Models are the standard scikit-learn estimators:
  - LogisticRegression (median-imputed, standardised, class-balanced)
  - HistGradientBoostingClassifier (native missing-value handling)
"""
import numpy as np

from sctc_core import make_folds


def _balanced_weights(y):
    y = np.asarray(y)
    pos = max((y == 1).sum(), 1)
    neg = max((y == 0).sum(), 1)
    return np.where(y == 1, neg / pos, 1.0)


def logreg_oof(X, y, stays, seed=0, k=5, C=1.0):
    """Out-of-fold logistic-regression predictions under stay-level folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    y = np.asarray(y).astype(int)
    fa = make_folds(stays, seed, k)
    pred = np.zeros(len(y))
    for f in range(k):
        tr, te = fa != f, fa == f
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=C, max_iter=1000))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def gbdt_oof(X, y, stays, seed=0, k=5, learning_rate=0.12, max_depth=3, max_iter=200):
    """Out-of-fold histogram gradient-boosting predictions under stay-level folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = np.asarray(y).astype(int)
    fa = make_folds(stays, seed, k)
    pred = np.zeros(len(y))
    for f in range(k):
        tr, te = fa != f, fa == f
        clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth, max_iter=max_iter,
            l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=_balanced_weights(y[tr]))
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred
