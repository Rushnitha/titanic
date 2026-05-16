"""
utils.py
--------
Shared helper functions used across the pipeline.
"""

import os
import json
import logging
from datetime import datetime

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that writes to stdout.

    Usage
    -----
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:                      # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ──────────────────────────────────────────
# File / Path helpers
# ──────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """
    Create *path* (and any missing parents) if it does not exist.
    Returns the path so callers can chain: open(ensure_dir(p) + '/f.csv').
    """
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> str:
    """Return the absolute path of the project root (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def data_path(*parts: str) -> str:
    """Build an absolute path inside data/  e.g. data_path('raw', 'titanic.csv')."""
    return os.path.join(project_root(), "data", *parts)


def models_path(*parts: str) -> str:
    """Build an absolute path inside models/."""
    return os.path.join(project_root(), "models", *parts)


# ──────────────────────────────────────────
# Metrics persistence
# ──────────────────────────────────────────

def save_metrics(metrics: dict, filename: str = "metrics.json") -> str:
    """
    Persist a metrics dictionary to  models/metrics.json  (or a custom name).

    Parameters
    ----------
    metrics  : dict   – any JSON-serialisable dictionary of results
    filename : str    – target filename inside models/

    Returns
    -------
    str – absolute path of the written file
    """
    dest = models_path(filename)
    ensure_dir(os.path.dirname(dest))
    metrics["saved_at"] = datetime.utcnow().isoformat()
    with open(dest, "w") as f:
        json.dump(metrics, f, indent=2)
    return dest


def load_metrics(filename: str = "metrics.json") -> dict:
    """Load a previously saved metrics JSON file. Returns empty dict if not found."""
    path = models_path(filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)