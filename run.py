#!/usr/bin/env python
"""
Main entry point for the Satellite Imagery Analysis web application.
Initializes and runs the Flask application.
"""

import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(PROJECT_ROOT) / 'app.log')
    ]
)
logger = logging.getLogger('satellite_app')

cache_dir = Path(PROJECT_ROOT) / 'cache'
cache_dir.mkdir(exist_ok=True)
logger.info(f"Cache directory: {cache_dir}")

from app.app import app

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Satellite Imagery Analysis web application on port {port}")
    logger.info(f"Debug mode: {debug}")
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting the application: {e}")
        sys.exit(1)