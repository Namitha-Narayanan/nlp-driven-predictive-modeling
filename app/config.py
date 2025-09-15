"""
Application configuration.

Defines default settings for the Flask app, pulled from environment
variables when available. Provides an `as_dict` method for convenient
loading into Flask's config.
"""

import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    """
    Configuration dataclass for the stateless prediction API.

    Attributes:
        HOST: Host address for the Flask server (default: 127.0.0.1).
        PORT: Port for the Flask server (default: 5000).
        DEBUG: Enable/disable debug mode (default: False).
        STRICT_SCHEMA: Whether to enforce strict schema validation (default: True).
        PREDICT_TIMEOUT: Timeout budget in seconds for predictions (default: 90.0).
        ROOT: Path to the app directory.
        PROJECT_ROOT: Path to the project root.
        DATA_DIR: Path to the static data directory.
    """
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    STRICT_SCHEMA: bool = os.getenv("STRICT_SCHEMA", "true").lower() == "true"
    PREDICT_TIMEOUT: float = float(os.getenv("PREDICT_TIMEOUT", "90.0"))  # keep  90s per README
 
    
    ROOT: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(ROOT)
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "static")

    def as_dict(self) -> dict:
        """
        Return configuration as a dictionary, suitable for Flask app.config.
        """
        return {
            "HOST": self.HOST,
            "PORT": self.PORT,
            "DEBUG": self.DEBUG,
            "STRICT_SCHEMA": self.STRICT_SCHEMA,
            "ROOT": self.ROOT,
            "PREDICT_TIMEOUT": self.PREDICT_TIMEOUT,
            "PROJECT_ROOT": self.PROJECT_ROOT,
            "DATA_DIR": self.DATA_DIR,
        }
