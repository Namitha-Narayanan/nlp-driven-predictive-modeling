#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PORT=5000
export DEBUG=true
export PREDICT_TIMEOUT=90.0

# Run the app
python main.py
