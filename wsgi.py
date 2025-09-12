#!/usr/bin/env python3
"""
WSGI entry point for the BIPV Heat Flask application.
This file is used by gunicorn and other WSGI servers for production deployment.
"""

import os
import sys
from pathlib import Path

# Add the application directory to the Python path
app_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(app_dir))

# Import the Flask application
from main import app

# Make sure the app is configured for production
if __name__ != "__main__":
    # Disable debug mode in production
    app.config['DEBUG'] = False
    # Set other production configurations as needed
    app.config['ENV'] = 'production'

# This is what gunicorn will look for
application = app

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    app.run(host='0.0.0.0', port=5000, debug=False)
