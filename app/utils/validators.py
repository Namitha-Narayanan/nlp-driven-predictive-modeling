"""
Validation utilities for request payloads.

Checks for:
- presence of required fields,
- type normalization (NumPy arrays, ints, str),
- shape consistency across observed/predict data.

Raises ValueError with clear messages when validation fails.
"""

from __future__ import annotations
import numpy as np

REQUIRED_FIELDS = ("x_observed", "y_observed", "x_predict", "t", "n", "k", "d")

def ensure_fields(payload: dict) -> None:
    """
    Ensure all required keys are present in the payload.

    Args:
        payload: Incoming request dictionary.

    Raises:
        ValueError: If any required field is missing.
    """
    missing = [k for k in REQUIRED_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

def _to_arr(x):
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=float)

def normalize(payload: dict) -> dict:
    """
    Normalize types and shapes for downstream processing.

    - Convert lists to NumPy arrays
    - Ensure y_observed is 1-D
    - Cast n, k, d to int
    - Cast t to str

    Args:
        payload: Raw request dictionary.

    Returns:
        dict: Normalized payload.
    """
    out = dict(payload)
    out["x_observed"] = _to_arr(out["x_observed"])
    out["y_observed"] = _to_arr(out["y_observed"]).reshape(-1)
    out["x_predict"]  = _to_arr(out["x_predict"])
    out["n"] = int(out["n"]); out["k"] = int(out["k"]); out["d"] = int(out["d"])
    out["t"] = str(out["t"])
    return out

def check_shapes(p: dict) -> None:
    """
    Validate array shapes against declared dimensions (n, k, d).

    Args:
        p: Normalized payload dictionary.

    Raises:
        ValueError: If any shape does not match the declared dimensions.
    """
    Xo, yo, Xp = p["x_observed"], p["y_observed"], p["x_predict"]
    n, k, d = p["n"], p["k"], p["d"]
    if Xo.shape != (n, d):
        raise ValueError(f"x_observed shape {Xo.shape} does not match (n,d)=({n},{d})")
    if yo.shape not in [(n,), (n, 1)]:
        raise ValueError(f"y_observed shape {yo.shape} must be (n,) or (n,1)")
    if Xp.shape != (k, d):
        raise ValueError(f"x_predict shape {Xp.shape} does not match (k,d)=({k},{d})")
