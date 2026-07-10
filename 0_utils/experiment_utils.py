"""
experiment_utils.py
-------------------
Utilities for reproducibility, timing, config management and result logging.

Typical usage
-------------
Use these helpers for reproducibility, timing, experiment logging, and config I/O.
"""

import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    print(f"Seed set to {seed}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple elapsed-time context manager.

    Example
    -------
    with Timer("model training"):
        model.fit(X_train, y_train)
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.label}] " if self.label else ""
        print(f"{label}Elapsed: {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load a JSON config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, path: str | Path) -> None:
    """Save a dict as a JSON config file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {path}")


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

class ResultLogger:
    """Accumulate experiment results and save to JSON or CSV.

    Example
    -------
    logger = ResultLogger()
    logger.log({"model": "CNN", "test_acc": 0.9927, "train_time": 52.3})
    logger.save("reports/results.json")
    logger.summary()
    """

    def __init__(self):
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        self.records.append(record)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(path).suffix.lower()

        if suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        elif suffix == ".csv":
            import pandas as pd
            pd.DataFrame(self.records).to_csv(path, index=False)
        else:
            raise ValueError("Unsupported format. Use .json or .csv")

        print(f"Results saved: {path}")

    def summary(self) -> None:
        """Print all logged records."""
        import pandas as pd
        if self.records:
            print(pd.DataFrame(self.records).to_string(index=False))
        else:
            print("No results logged yet.")


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def print_versions() -> None:
    """Print versions of common data science packages."""
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn",
                "tensorflow", "torch", "scipy"]
    for name in packages:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"  {name}: {version}")
        except ImportError:
            pass
