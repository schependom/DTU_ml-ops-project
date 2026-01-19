import datetime
import json
import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

import wandb
from ml_ops_project.models import SentimentClassifier

load_dotenv()
# Define model and device configuration
BUCKET_NAME = "ml_ops_project_g7"  # Used for saving predictions
MODEL_FILE_NAME = "models/best_model.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReviewInput(BaseModel):
    """Define input data structure for the endpoint."""

    review: str


class PredictionOutput(BaseModel):
    """Define output data structure for the endpoint."""

    sentiment: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    global model, class_names

    # Check if model exists, if not download
    if not os.path.exists(MODEL_FILE_NAME):
        download_model_from_wandb()

    # Load from PTL checkpoint
    # We assume usage of the model definition from ml_ops_project.models
    # which is a LightningModule.
    print(f"Loading model from {MODEL_FILE_NAME}...")
    model = SentimentClassifier.load_from_checkpoint(MODEL_FILE_NAME, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Tokenizer is handled inside the SentimentClassifier (self.tokenizer)

    # Based on training logic (Rotten Tomatoes), we have 2 classes
    class_names = ["negative", "positive"]
    print("Model and tokenizer loaded successfully")

    yield

    del model


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


def download_model_from_wandb():
    """Download the best model from W&B Model Registry."""
    print("Downloading model from W&B...")
    api = wandb.Api()

    # --- CONFIGURATION START ---
    # The organization name where the registry lives
    entity = os.getenv("WANDB_ORGANIZATION")

    # The name of the registry (which acts as a W&B Project)
    project = "wandb-registry-model_registry"

    # The specific collection name you requested
    collection = "sentiment_classifier_models"

    # The version alias
    alias = "inference"
    # --- CONFIGURATION END ---

    # Construct the full path: entity/project/collection:alias
    artifact_path = f"{entity}/{project}/{collection}:{alias}"

    print(f"Attempting to fetch artifact from: {artifact_path}")

    try:
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download()

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        # Find the checkpoint file in the downloaded artifacts
        for file in os.listdir(artifact_dir):
            if file.endswith(".ckpt"):
                source = os.path.join(artifact_dir, file)
                import shutil

                shutil.move(source, MODEL_FILE_NAME)
                print(f"Model downloaded to {MODEL_FILE_NAME}")
                return

        raise RuntimeError("No .ckpt file found in W&B artifact")

    except Exception as e:
        print(f"Failed to download model from W&B: {e}")
        if "not found" in str(e):
            #
            print(
                f"HINT: Check permissions for 'MLOpsServiceAccount' in 'turtle_team-org'. "
                f"Ensure the artifact '{artifact_path}' exists."
            )
        raise


# Save prediction results to GCP (keeping this as requested implicitly, only changing download)
def save_prediction_to_gcp(review: str, outputs: list[float], sentiment: str):
    """Save the prediction results to GCP bucket."""
    from google.cloud import storage  # Import here since we removed top-level import

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)
    # Prepare prediction data
    data = {
        "review": review,
        "sentiment": sentiment,
        "probability": outputs,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    blob = bucket.blob(f"predictions/prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review_input: ReviewInput, background_tasks: BackgroundTasks):
    """Predict sentiment of the input text."""
    try:
        # Tokenize using the model's tokenizer
        encoding = model.tokenizer(
            review_input.review,
            add_special_tokens=True,
            max_length=160,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,  # Added truncation for safety
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Model prediction
        with torch.no_grad():
            # PTL model forward returns a SequenceClassifierOutput
            # We access .logits from it
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            _, prediction = torch.max(logits, dim=1)
            sentiment = class_names[prediction]

        background_tasks.add_task(save_prediction_to_gcp, review_input.review, probs.tolist()[0], sentiment)

        return PredictionOutput(sentiment=sentiment)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
