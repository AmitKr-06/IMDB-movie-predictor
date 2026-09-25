import json
import mlflow
import logging
from src.logger import logging
import os
from dotenv import load_dotenv
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Below code block is for Production use
# -----------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking

dagshub_token = os.getenv("IMDB_Rating_TEST")
if not dagshub_token:
    raise EnvironmentError("IMDB_Rating_TEST environment vaiable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AmitKr-06"
repo_name = "IMDB-movie-predictor"
# Set up MLFlow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# ----------------------------------------------------------------------------------------

# Below code block is for local use
# --------------------------------------------------------
# mlflow.set_tracking_uri("pass link of dagshub_uri")
# dagshub.init("Dagshub.init link")

def load_model_info(file_path: str) -> dict:
    """ Load the model info from a JSON file. """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """ Register the model to the MLflow Model Registry."""
    try:
        client = mlflow.tracking.MlflowClient()

        # Ensure the registered model exists (skip if it's already there)
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass

        # Register the version directly against the known artifact path,
        # bypassing mlflow.register_model()'s internal artifact lookup
        # (that lookup fails against DagsHub's tracking backend)
        source = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = client.create_model_version(
            name = model_name,
            source = source,
            run_id = model_info['run_id']
        )

        # Set an alias instead of the deprecated stage-transition API
        client.set_registered_model_alias(
            name = model_name,
            alias = "staging",
            version = model_version.version
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and aliased as "staging".')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()