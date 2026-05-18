from flask import Flask
from flask_cors import CORS
from api.routes import init_routes
from api.data_db import create_tables
import subprocess
import logging

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    init_routes(app)

    create_tables()
    
    enable_git_longpaths()

    return app


# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("git_longpaths.log"),  # Καταγραφή σε αρχείο
        logging.StreamHandler(),  # Καταγραφή και στην κονσόλα
    ],
)

def enable_git_longpaths():
    try:
        logging.info("Enabling Git long paths support...")
        result = subprocess.run(
            ["git", "config", "--system", "core.longpaths", "true"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info("Git long paths support enabled successfully.")
        logging.debug(f"Command output: {result.stdout.decode().strip()}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error enabling Git long paths: {e.stderr.decode().strip()}")
    except PermissionError:
        logging.critical("Permission denied: Try running the script as administrator.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")


