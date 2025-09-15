from __future__ import annotations
import numpy as np

def shape_str(a, name: str = "array") -> str:
    arr = np.asarray(a)
    return f"{name}: {arr.dtype}{tuple(arr.shape)}"
