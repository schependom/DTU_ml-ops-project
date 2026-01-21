import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from hydra import compose, initialize
from pydantic import BaseModel

load_dotenv()

import wandb
from ml_ops_project.models import SentimentClassifier

ml_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wandb_setup(path: str):
    wandb.login(key=os.getenv("WANDB_SERVICE_USER"))

    organization = os.getenv("WANDB_ORGANIZATION")
    run = wandb.init(project="MLOps-Project", job_type="inference")

    artifact_path = organization + "/" + path
    # entity/registry/collection:alias
    artifact = run.use_artifact(artifact_path, type="model")

    model_dir = artifact.download()
    return model_dir


def load_model():
    with initialize(version_base="1.2", config_path="../../configs"):
        # Load the config object
        cfg = compose(config_name="config")
    path = f"{cfg.inference.registry}/{cfg.inference.collection}:{cfg.inference.alias}"
    model_dir = wandb_setup(path)

    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    model = SentimentClassifier.load_from_checkpoint(checkpoint_path, model_name=cfg.model.name, weights_only=False)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = load_model()
    ml_models["tokenizer"] = ml_models["model"].tokenizer

    yield

    ml_models.clear()
    print("Cleaning up model resources...")


app = FastAPI(lifespan=lifespan)


# input schema
class InferenceInput(BaseModel):
    statement: str


class InferenceOutput(BaseModel):
    sentiment: int


# inference endpoint
@app.post("/inference", response_model=InferenceOutput)
async def predict(data: InferenceInput):
    try:
        model = ml_models.get("model")
        tokenizer = ml_models.get("tokenizer")

        if not model:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Perform inference
        input = tokenizer(data.statement, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            logits = model(**input).logits
            prediction_id = torch.argmax(logits, dim=1).item()

        return InferenceOutput(sentiment=prediction_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
