"""
Response helpers for the API.

All endpoints return JSON in a consistent format:
- success(...) -> {"status": "success", "predictions": [...]}
- client_error(...) -> {"status": "error", "message": "..."} with 4xx
- server_error(...) -> {"status": "error", "message": "..."} with 5xx
- timeout_error(...) -> {"status": "error", "message": "Request timeout"} with 408

This keeps the contract predictable for both humans and the test harness.
"""

from flask import jsonify

# 200 OK — successful prediction
def success(predictions, **meta):
    
    # Coerce to plain floats for JSON (handles numpy types too)
    preds = [float(x) for x in predictions]
    payload = {"status": "success", "predictions": preds}
    if meta:
        payload["meta"] = meta
    return jsonify(payload), 200

# 400 Bad Request — user-side error (e.g. missing fields, bad shapes)
def client_error(message: str, *, status_code: int = 400, **extra):
    payload = {"status": "error", "message": str(message)}
    if extra:
        payload.update(extra)
    return jsonify(payload), status_code

# 500 Internal Server Error — unexpected failure in model or server logic
def server_error(message: str, *, status_code: int = 500, **extra):
    payload = {"status": "error", "message": str(message)}
    if extra:
        payload.update(extra)
    return jsonify(payload), status_code


# 408 Request Timeout — when a request exceeds the allowed time budget i.e 90 seconds
def timeout_error(message: str = "Request timeout", **extra):
    """
    App-level timeout budget breached 
    """
    payload = {"status": "error", "message": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), 408
