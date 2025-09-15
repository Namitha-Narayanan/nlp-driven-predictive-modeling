"""
Simple parser for natural-language hints.

Takes short phrases like:
- "linear in x3"
- "quadratic in x2"
- "periodic in x1"

and turns them into a structured recipe (kind, degree, dimension, harmonics).
The degrees/harmonics are capped conservatively so the model stays stable on tiny data.
"""

from __future__ import annotations

def parse_hint(t: str) -> dict:
    """
    Parse a natural-language hint into a feature recipe.

    Args:
        t (str): Text description of the relationship, e.g.
                 "quadratic in x2", "periodic in x1 with two peaks".

    Returns:
        dict: {
            "kind": str   -> one of {"linear", "poly", "periodic"},
            "degree": int -> polynomial degree (1–3),
            "dim": int    -> which x dimension is referenced (1–3),
            "harmonics": int -> number of sinusoidal harmonics (1–3)
        }
    """
    s = (t or "").lower()

    # Default values
    kind = "linear"
    degree = 1
    dim = 3
    harmonics = 1

    # Detect which x-dimension the hint refers to
    if "x1" in s: dim = 1
    if "x2" in s: dim = 2
    if "x3" in s: dim = 3

    # Detect polynomial degree
    if "quadratic" in s or "polynomial" in s:
        kind, degree = "poly", 2
    if "cubic" in s:
        kind, degree = "poly", 3

    # Detect periodicity (sinusoids, oscillations, waves)
    if any(w in s for w in ["periodic", "sinusoid", "oscillation", "wave"]):
        kind = "periodic"
        if "high" in s or "three peak" in s or "three peaks" in s:
            harmonics = 3
        elif "two" in s or "2 peak" in s:
            harmonics = 2
        else:
            harmonics = 1

     # Heuristic: vague "flat then rising" hints → allow quadratic
    if "flat" in s and "then" in s and kind == "linear":
        # allow curvature for vague piecewise hints
        kind, degree = "poly", 2

    # Safety caps to avoid runaway complexity
    degree = max(1, min(degree, 3))
    harmonics = max(1, min(harmonics, 3))

    return {"kind": kind, "degree": degree, "dim": dim, "harmonics": harmonics}
