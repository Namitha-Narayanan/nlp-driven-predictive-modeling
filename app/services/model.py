"""
Core prediction path for the stateless API.

Parses the natural-language hint, builds consistent design matrices for
observed and future inputs, fits a stable ridge model, and returns
deterministic predictions. Kept lightweight and fast by design.
"""

from __future__ import annotations

from typing import List
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeCV, Ridge  # both available; we’ll prefer RidgeCV

from .text_hints import parse_hint
from .basis import build_design_matrix, transform_design_matrix

FloatArray = NDArray[np.float64]

def predict_from_payload(data: dict) -> List[float]:
    """
    Run the full prediction pipeline on a single request payload.

    Steps:
      1) Parse the text hint `t` into a feature recipe.
      2) Convert input lists to NumPy arrays (with explicit dtypes/shapes).
      3) Build design matrices for observed/predict using the same recipe.
      4) Fit a ridge model (CV for alpha when available) on observed data.
      5) Predict for `x_predict` and return plain Python floats.

    Args:
        data: Dict containing keys:
              - "x_observed": list[list[float]]
              - "y_observed": list[float]
              - "x_predict":  list[list[float]]
              - "t":          str (natural-language hint)

    Returns:
        list[float]: Predictions for the rows in `x_predict`, in order.

    Notes:
        Ridge regularization improves stability for small n and correlated
        features. The pipeline is deterministic and avoids randomness.
    """
    # Convert JSON lists -> NumPy for numeric stability and vectorization
    Xo: FloatArray = np.asarray(data["x_observed"], dtype=float)
    yo: FloatArray = np.asarray(data["y_observed"], dtype=float).reshape(-1)
    Xp: FloatArray = np.asarray(data["x_predict"],  dtype=float)
    t: str = str(data["t"])

    # Hint guides which basis functions we generate
    hint = parse_hint(t)

    # Build design matrices and standardize (fit on observed, apply to predict)
    Phi_obs, scaler, _meta = build_design_matrix(Xo, hint, fit=True)
    Phi_pred: FloatArray = transform_design_matrix(Xp, hint, scaler)

    # ---- Fail early with helpful messages (shape sanity) --------------------
    if Phi_obs.ndim != 2:
        raise ValueError(f"Expected 2D Phi_obs, got {Phi_obs.ndim}D with shape {Phi_obs.shape}")
    if Phi_pred.ndim != 2:
        raise ValueError(f"Expected 2D Phi_pred, got {Phi_pred.ndim}D with shape {Phi_pred.shape}")
    if Phi_obs.shape[0] != yo.shape[0]:
        raise ValueError(f"Rows of Phi_obs ({Phi_obs.shape[0]}) must match len(y_observed) ({yo.shape[0]})")

    # Tiny alpha grid (1D array-like). Explicit dtype for clarity.
    alphas = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0], dtype=float)

    try:
        model = RidgeCV(alphas=alphas)
    except Exception:
        # Extremely defensive: fall back if RidgeCV import/behavior ever fails
        model = Ridge(alpha=1.0)

    # Fit and predict
    model.fit(Phi_obs, yo)
    y_hat: FloatArray = np.asarray(model.predict(Phi_pred), dtype=float)
    
     # Plain floats keep JSON serialization straightforward
    return y_hat.tolist()
