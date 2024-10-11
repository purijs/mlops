# fastapi/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ray
import os
import logging
from typing import Dict

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Ray using environment variable
RAY_ADDRESS = os.getenv("RAY_ADDRESS")
if not RAY_ADDRESS:
    logger.error("RAY_ADDRESS environment variable is not set.")
    raise ValueError("RAY_ADDRESS environment variable is not set.")

try:
    ray.init(address=RAY_ADDRESS)
    logger.debug(f"Connected to Ray at address: {RAY_ADDRESS}")
except Exception as e:
    logger.error(f"Failed to connect to Ray: {e}")
    raise

class TrainingRequest(BaseModel):
    hyperparameters: Dict[str, float]  # Specify types as needed

@ray.remote
def train_model(hyperparameters: Dict[str, float]):
    """
    This function runs on Ray workers.
    """
    logger.debug(f"Training with hyperparameters: {hyperparameters}")

    # Example: Save the model to MinIO
    # from minio import Minio
    # minio_client = Minio(
    #     os.getenv("MINIO_ENDPOINT"),
    #     access_key=os.getenv("MINIO_ACCESS_KEY"),
    #     secret_key=os.getenv("MINIO_SECRET_KEY"),
    #     secure=False
    # )
    # model_data = ...  
    # minio_client.put_object("models", "latest_model.pkl", model_data, len(model_data))

    model_id = "model-123"  # Replace with actual model ID or identifier
    logger.debug(f"Training completed. Model ID: {model_id}")
    return model_id

@app.post("/train")
def trigger_training(request: TrainingRequest):
    """
    Endpoint to trigger a training job with specified hyperparameters.
    """
    logger.debug(f"Received training request: {request.hyperparameters}")
    try:
        # Submit training job to Ray
        model_id_future = train_model.remote(request.hyperparameters)
        model_id = ray.get(model_id_future)  # Optionally, handle asynchronously
        logger.debug(f"Submitted training job. Model ID: {model_id}")
        return {"model_id": model_id}
    except Exception as e:
        logger.error(f"Error during training job submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/latest")
def get_latest_model():
    """
    Endpoint to retrieve the latest trained model.
    """
    logger.debug("Retrieving the latest model from MinIO")
    try:
        # TODO: Implement retrieval logic from MinIO
        # Example:
        # minio_client = Minio(
        #     os.getenv("MINIO_ENDPOINT"),
        #     access_key=os.getenv("MINIO_ACCESS_KEY"),
        #     secret_key=os.getenv("MINIO_SECRET_KEY"),
        #     secure=False
        # )
        # response = minio_client.get_object("models", "latest_model.pkl")
        # model_data = response.read()
        # return {"model_data": model_data}
        latest_model_id = "model-123"  # Replace with actual retrieval logic
        return {"latest_model_id": latest_model_id}
    except Exception as e:
        logger.error(f"Error retrieving the latest model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

