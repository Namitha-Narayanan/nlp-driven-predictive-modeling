"""
Health check endpoint.

Endpoint: GET /health just returns {"status": "ok"}.
This is mainly used by the smoke tests to make sure the service is alive.
"""


from flask import Blueprint, jsonify
bp = Blueprint("health", __name__)

@bp.get("/health")
def health():
    return jsonify({"status": "ok"}), 200
