"""
App package initializer.

Defines the Flask application factory and registers routes, CORS, and static file handling.
"""


from flask import Flask, send_from_directory
from flask_cors import CORS
from .config import AppConfig

import os

def create_app(config_object: type[AppConfig] | None = None) -> Flask:
    """
    Application factory for the stateless prediction API.

    - Creates and configures the Flask app
    - Enables CORS
    - Loads configuration from AppConfig
    - Registers blueprints for health and predict endpoints
    - Serves static assets and frontend.html

    Args:
        config_object: Optional AppConfig class to override defaults.

    Returns:
        Flask: Configured Flask application instance.
    """
    
    app = Flask(__name__)
    CORS(app)

    cfg = (config_object or AppConfig)()
    app.config.from_mapping(cfg.as_dict())

    # Register blueprints
    from .routes.health import bp as health_bp
    from .routes.predict import bp as predict_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)
    
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STATIC_DIR = os.path.join(ROOT, "static")

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(STATIC_DIR, filename)

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "frontend.html")

    return app
