"""
Feature engineering utilities.

Generates design matrices from raw inputs using hints about the
functional form. Supports linear, polynomial, and periodic bases.
Outputs are standardized to improve numerical stability in regression.
"""
from __future__ import annotations
from typing import Tuple, Dict, Union
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FloatArray = NDArray[np.float64]
Meta = Dict[str, Union[str, int]]

def _fourier_1d(x: FloatArray, dim: int, harmonics: int, w: float = 1.0) -> FloatArray:
    """
    Construct a one-dimensional Fourier basis.

    Args:
        x: Input array of shape (n, d).
        dim: Dimension index to expand (1-based).
        harmonics: Number of harmonics to include.
        w: Base frequency multiplier.

    Returns:
        Array of shape (n, m) with [x_dim, sin(w*x_dim), cos(w*x_dim), ...].
    """
    j = dim - 1 # convert from 1-based to 0-based indexing
    v = x[:, [j]]
    feats = [v]
    for h in range(1, harmonics + 1):
        feats.append(np.sin(h * w * v))
        feats.append(np.cos(h * w * v))
    return np.asarray(np.hstack(feats), dtype=float)

def _poly_features(X: FloatArray, degree: int, fit: bool) -> FloatArray:
    """
    Generate polynomial features.

    Args:
        X: Input array of shape (n, d).
        degree: Maximum degree of polynomial terms.
        fit: Whether to fit the transformer or just apply it.

    Returns:
        Dense float64 array of polynomial features.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Phi = poly.fit_transform(X) if fit else poly.transform(X)
    Phi = np.asarray(Phi, dtype=float)
    return Phi

def build_design_matrix(X: FloatArray, hint: Dict, fit: bool = True
                       ) -> Tuple[FloatArray, StandardScaler, Meta]:
    """
    Build and standardize a design matrix from raw inputs.

    Args:
        X: Input array of shape (n, d).
        hint: Dictionary describing feature recipe.
        fit: If True, fit the scaler on this data.

    Returns:
        Tuple of:
            - Phi_s: Standardized features, shape (n, p).
            - scaler: Fitted StandardScaler instance.
            - meta: Dictionary of the applied recipe.
    """
    X = np.asarray(X, dtype=float)

    kind = hint.get("kind", "linear")
    degree = int(hint.get("degree", 1))
    dim = int(hint.get("dim", 3))
    harmonics = int(hint.get("harmonics", 1))

    if kind == "linear":
        Phi = X
    elif kind == "poly":
        Phi = _poly_features(X, degree=degree, fit=fit)
    elif kind == "periodic":
        F = _fourier_1d(X, dim=dim, harmonics=harmonics, w=1.0)
        Phi = np.hstack([X, F])
    else:
        Phi = X

    scaler = StandardScaler(with_mean=True, with_std=True)
    Phi_s = scaler.fit_transform(Phi) if fit else scaler.transform(Phi)
    Phi_s = np.asarray(Phi_s, dtype=float)
    
    meta: Meta = {"kind": str(kind), "degree": degree, "dim": dim, "harmonics": harmonics}
    return Phi_s, scaler, meta

def transform_design_matrix(X: FloatArray, hint: Dict, scaler: StandardScaler) -> FloatArray:
    """
    Transform new inputs into standardized features using a fitted recipe.

    Args:
        X: Input array of shape (k, d).
        hint: Dictionary describing feature recipe (must match training).
        scaler: StandardScaler fitted on observed data.

    Returns:
        Standardized feature matrix of shape (k, p).
    """
    X = np.asarray(X, dtype=float)

    kind = hint.get("kind", "linear")
    degree = int(hint.get("degree", 1))
    dim = int(hint.get("dim", 3))
    harmonics = int(hint.get("harmonics", 1))

    if kind == "linear":
        Phi = X
    elif kind == "poly":
        Phi = _poly_features(X, degree=degree, fit=False)
    elif kind == "periodic":
        F = _fourier_1d(X, dim=dim, harmonics=harmonics, w=1.0)
        Phi = np.hstack([X, F])
    else:
        Phi = X

    Phi_s = scaler.transform(Phi)
    return np.asarray(Phi_s, dtype=float)
