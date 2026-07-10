"""
split_utils.py
--------------
General-purpose train/val/test splitting utilities.
Covers random stratified splits and time-order splits.

Typical usage
-------------
Use these helpers for random, stratified, k-fold, and time-aware data splits.
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


# ---------------------------------------------------------------------------
# Random splits (classification / regression)
# ---------------------------------------------------------------------------

def random_split(
    X,
    y,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple:
    """Two-step split into train / val / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    strat = y if stratify else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state,
    )
    val_frac = val_size / (1.0 - test_size)
    strat2 = y_tv if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=strat2, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_test_only(
    X,
    y,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple:
    """Simple train/test split (no validation set).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Time-order split (time series — never shuffle)
# ---------------------------------------------------------------------------

def time_split(
    X,
    y,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple:
    """Split by time order: no shuffle, no leakage.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    return (
        X[:train_end], X[train_end:val_end], X[val_end:],
        y[:train_end], y[train_end:val_end], y[val_end:],
    )


# ---------------------------------------------------------------------------
# Cross-validation fold generators
# ---------------------------------------------------------------------------

def kfold_splits(
    X,
    y,
    n_splits: int = 5,
    stratify: bool = True,
    random_state: int = 42,
):
    """Yield (X_train, X_val, y_train, y_val) for each CV fold.

    Example
    -------
    for X_tr, X_v, y_tr, y_v in kfold_splits(X, y, n_splits=5):
        model.fit(X_tr, y_tr)
    """
    kf = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    for train_idx, val_idx in kf.split(X, y):
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def split_summary(y_train, y_val, y_test=None) -> None:
    """Print class distribution across splits."""
    import pandas as pd

    def _dist(y, name):
        s = pd.Series(y).value_counts(normalize=True).sort_index() * 100
        print(f"  {name} ({len(y)} samples): " +
              ", ".join(f"{k}={v:.1f}%" for k, v in s.items()))

    print("\nSplit summary:")
    _dist(y_train, "train")
    _dist(y_val,   "val  ")
    if y_test is not None:
        _dist(y_test, "test ")
