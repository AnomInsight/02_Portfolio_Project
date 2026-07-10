"""
io_utils.py
-----------
General file I/O utilities: save/load models, configs, metrics, predictions.

Typical usage
-------------
Use these helpers for JSON/CSV/Parquet/model artifact save-load workflows.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str | Path) -> None:
    """Save a dict to JSON. Handles numpy types automatically."""
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_Encoder)
    print(f"Saved JSON: {path}")


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV / Parquet
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=index)
    print(f"Saved CSV: {path}")


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)
    print(f"Saved Parquet: {path}")


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# NumPy arrays
# ---------------------------------------------------------------------------

def save_arrays(path: str | Path, **arrays: np.ndarray) -> None:
    """Save one or more arrays as a .npz file.

    Example: save_arrays("data/splits.npz", X_train=X_train, y_train=y_train)
    """
    ensure_dir(Path(path).parent)
    np.savez(path, **arrays)
    print(f"Saved arrays: {path}")


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load arrays from a .npz file into a dict."""
    return dict(np.load(path))


# ---------------------------------------------------------------------------
# Models (pickle / sklearn / keras)
# ---------------------------------------------------------------------------

def save_pickle(obj, path: str | Path) -> None:
    """Save any Python object (sklearn models, etc.) with pickle."""
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved pickle: {path}")


def load_pickle(path: str | Path):
    """Load a pickled object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_keras_model(model, path: str | Path) -> None:
    """Save a Keras model in native format."""
    ensure_dir(Path(path).parent)
    model.save(str(path))
    print(f"Saved Keras model: {path}")


def load_keras_model(path: str | Path):
    """Load a Keras model from disk."""
    from tensorflow import keras
    return keras.models.load_model(str(path))
