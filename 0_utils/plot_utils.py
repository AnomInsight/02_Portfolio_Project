"""
plot_utils.py
-------------
General-purpose plotting utilities for classification, regression,
feature analysis, and distribution inspection.

Typical usage
-------------
Use these helpers for class balance, correlation, and distribution visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: str, dpi: int = 200) -> None:
    """Save a figure to disk with tight layout."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved: {path}")


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y,
    title: str = "Class Distribution",
    save_path: str | None = None,
) -> None:
    """Bar chart of class label counts."""
    s = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(k) for k in s.index], s.values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: str | None = None,
) -> None:
    """Heatmap of pairwise Pearson correlations."""
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr)), max(6, len(corr) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_histograms(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    bins: int = 30,
    save_path: str | None = None,
) -> None:
    """Grid of histograms for numeric columns."""
    cols = cols or df.select_dtypes(include="number").columns.tolist()
    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flat
    for ax, col in zip(axes, cols):
        df[col].hist(bins=bins, ax=ax)
        ax.set_title(col)
    for ax in list(axes)[n:]:
        ax.axis("off")
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing Values",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of missing value percentages."""
    missing_pct = df.isnull().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        print("No missing values found.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing_pct) * 0.4)))
    missing_pct.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("% Missing")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Annotated heatmap confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or "auto",
                yticklabels=labels or "auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_learning_curves(
    history: dict,
    title: str = "Learning Curves",
    save_path: str | None = None,
) -> None:
    """Train/val loss and accuracy curves from a Keras history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 20,
    title: str = "Feature Importances",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of top-n feature importances."""
    idx = np.argsort(importances)[-top_n:][::-1]
    labels = [feature_names[i] for i in idx] if feature_names else [str(i) for i in idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[idx])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: str | None = None,
) -> None:
    """Residual scatter and distribution for regression tasks."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual vs Predicted")

    axes[1].hist(residuals, bins=40)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()
