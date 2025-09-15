# """
# Prediction endpoint (POST /predict).

# This is the main logic:
# 1. Validate the incoming JSON payload
# 2. Parse the natural-language hint into a feature spec
# 3. Build design matrices for observed and future inputs
# 4. Fit a ridge regression model on the observed data
# 5. Predict values for x_predict and return them as JSON

# If the input text is invalid(bad input, wrong shape) or the service is interrupted, a clear error message along with http code is returned.
# """

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
from flask import Blueprint, request

from app.utils.responses import (
    success, client_error, server_error, timeout_error
)

bp = Blueprint("predict", __name__)

@bp.post("/predict")
def predict():
    # -------- read JSON --------
    payload = request.get_json(force=True)
    start_time = time.perf_counter()

    # # -------- test switches for screenshots (ONLY if these flags are included) --------
    # # For forced timeout or failure modes i
    # if payload.get("force_timeout") is True:
    #     time.sleep(95)                 # SLA is 90s
    #     return timeout_error()         # -> 408 {"status":"error","message":"Request timeout"}

    # if payload.get("force_fail") is True:
    #     # for screenshot only; normal failures are handled below with real message
    #     return server_error("Linear regression failed")  # -> as per sample test load from readme.md file
    # # --------------------------------------------------------------------------------

    # -------- validation (400s) --------
    required = ["x_observed", "y_observed", "x_predict", "t", "n", "k", "d"]
    for key in required:
        if key not in payload:
            
            return client_error(f"Missing required field: '{key}'")

    # ints
    try:
        n = int(payload["n"]); k = int(payload["k"]); d = int(payload["d"])
    except Exception:
        return client_error("Invalid input: 'n', 'k', and 'd' must be integers")

    # arrays
    try:
        x_obs = np.array(payload["x_observed"], dtype=float)
        y_obs = np.array(payload["y_observed"], dtype=float)
        x_pred = np.array(payload["x_predict"], dtype=float)
    except Exception as e:
        return client_error("Invalid input: non-numeric values encountered", detail=str(e))

    # shapes 
    exp_x_obs = (n, d)
    exp_y_obs = (n,)
    exp_x_pred = (k, d)

    if x_obs.shape != exp_x_obs:
        return client_error(
            f"Invalid input: x_observed shape {x_obs.shape} doesn't match expected {exp_x_obs}"
        )
    if y_obs.shape != exp_y_obs:
        return client_error(
            f"Invalid input: y_observed shape {y_obs.shape} doesn't match expected {exp_y_obs}"
        )
    if x_pred.shape != exp_x_pred:
        return client_error(
            f"Invalid input: x_predict shape {x_pred.shape} doesn't match expected {exp_x_pred}"
        )

    # -------- modelling (wrapped so the 90s SLA is enforced) --------
    def _run_model():
        """
        Replace this baseline with your real pipeline if you have it:
        - parse text hint
        - build basis/design matrix
        - fit model
        - predict
        """
        # Baseline: intercept + linear least squares
        X_obs = np.c_[np.ones((n, 1)), x_obs]
        X_pred2 = np.c_[np.ones((k, 1)), x_pred]
        coeffs, *_ = np.linalg.lstsq(X_obs, y_obs, rcond=None)
        return X_pred2 @ coeffs

    # Remaining time in the 90s budget
    elapsed = time.perf_counter() - start_time
    remaining = max(0.0, 90.0 - elapsed)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_model)
            preds = fut.result(timeout=remaining)   # <- hard cap to meet README SLA
    except FuturesTimeout:
        # Exceeded the total 90s budget → 408
        return timeout_error()
    except Exception as e:
        # Any other failure → 500
        return server_error(str(e), error_type=type(e).__name__)

    # -------- success --------
    return success(np.asarray(preds).tolist())
