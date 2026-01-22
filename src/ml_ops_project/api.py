"""FastAPI inference service for sentiment classification.

This module provides:
1. Model loading via W&B artifacts with Hydra config
2. Prometheus metrics for requests, latency, and errors
3. An inference endpoint that returns a sentiment label
4. Optional persistence of predictions to a GCP bucket
"""

import datetime
import json
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage
from hydra import compose, initialize
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
from pydantic import BaseModel

import wandb
from ml_ops_project.models import SentimentClassifier

load_dotenv()

# Cache for loaded model and tokenizer to avoid reloads per request
ml_models = {}
# Select the best available device for inference
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define Prometheus metrics for request observability
error_counter = Counter("prediction_error", "Number of prediction errors")
request_counter = Counter("prediction_requests", "Number of prediction requests")
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
review_summary = Summary("review_length_summary", "Review length summary")


# Load model artifacts from W&B to a local directory
def wandb_setup(path: str):
    wandb.login(key=os.getenv("WANDB_SERVICE_USER"))

    organization = os.getenv("WANDB_ORGANIZATION")
    project = os.getenv("WANDB_PROJECT")
    run = wandb.init(project=project, job_type="inference")

    artifact_path = organization + "/" + path
    artifact = run.use_artifact(artifact_path, type="model")

    model_dir = artifact.download()
    return model_dir


# Build the model from the W&B checkpoint referenced in config
def load_model(cfg):
    path = f"{cfg.inference.registry}/{cfg.inference.collection}:{cfg.inference.alias}"
    model_dir = wandb_setup(path)

    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    model = SentimentClassifier.load_from_checkpoint(checkpoint_path, model_name=cfg.model.name, weights_only=False)
    return model


# Save prediction results to a GCP bucket for auditability
def save_prediction_to_gcp(review: str, outputs: list[float], sentiment: int, bucket_name: str):
    """Save the prediction results to GCP bucket."""

    client = storage.Client()
    print(client)
    bucket = client.bucket(bucket_name)
    print(bucket_name)
    print(bucket)
    time = datetime.datetime.now(tz=datetime.timezone.utc)
    # Prepare prediction data
    data = {
        "review": review,
        "sentiment": sentiment,
        "probability": outputs,
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }
    blob = bucket.blob(f"predictions/prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


# Make Hydra config available globally for startup and requests
with initialize(version_base="1.2", config_path="../../configs"):
    # Load the config object
    cfg = compose(config_name="config")


# Startup and shutdown lifecycle: load model once, clear on exit
@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_model(cfg)
    model.to(device)
    model.eval()
    ml_models["model"] = model
    ml_models["tokenizer"] = model.tokenizer

    yield

    ml_models.clear()
    print("Cleaning up model resources...")


# Create the FastAPI app and expose Prometheus metrics
app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


# Request/response schemas for the inference endpoint
class InferenceInput(BaseModel):
    statement: str


class InferenceOutput(BaseModel):
    sentiment: int


# Inference endpoint: validate input, run model, enqueue GCP write
@app.post("/inference", response_model=InferenceOutput)
async def predict(data: InferenceInput, background_tasks: BackgroundTasks):
    # Update Prometheus metrics
    request_counter.inc()

    # Measure latency
    with request_latency.time():
        try:
            model = ml_models.get("model")
            tokenizer = ml_models.get("tokenizer")

            if not model:
                raise HTTPException(status_code=500, detail="Model not loaded")

            # Perform inference
            input = tokenizer(data.statement, return_tensors="pt", padding=True, truncation=True)
            input = {key: value.to(device) for key, value in input.items()}

            with torch.no_grad():
                logits = model(**input).logits
                probabilities = F.softmax(logits, dim=1).detach().cpu()
                prediction_id = torch.argmax(logits, dim=1).item()

            background_tasks.add_task(
                save_prediction_to_gcp,
                data.statement,
                probabilities.tolist()[0],
                prediction_id,
                cfg.cloud.bucket_name,
            )

            return InferenceOutput(sentiment=prediction_id)

        except Exception as e:
            # Increment error counter
            error_counter.inc()

            raise HTTPException(status_code=500, detail=str(e)) from e
