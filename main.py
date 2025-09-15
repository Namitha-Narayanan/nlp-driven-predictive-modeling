"""
Main entry point for the API.

Here the code is designed to spin up a small Flask server, plug in to the expected routes,
and enable CORS so the frontend can talk to it.
"""
from app import create_app
from app.config import AppConfig

app = create_app()

if __name__ == "__main__":
    cfg = AppConfig()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
