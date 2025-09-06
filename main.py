import logging
import os

# Configure logging for better debugging
logging.basicConfig(level=logging.DEBUG)

try:
    from app import app
    logging.info("Flask app imported successfully")
except Exception as e:
    logging.error(f"Failed to import Flask app: {e}")
    raise

# Ensure the Flask app is properly accessible for Gunicorn deployment
# This is the main entry point that Gunicorn will use
def create_app():
    """Application factory for deployment"""
    try:
        logging.info("Creating Flask application for deployment...")
        return app
    except Exception as e:
        logging.error(f"Failed to create Flask application: {e}")
        raise

# Ensure the Flask app is properly accessible for deployment
if __name__ == '__main__':
    try:
        logging.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"Failed to start Flask application: {e}")
        raise

# Export the app for Gunicorn
application = app
