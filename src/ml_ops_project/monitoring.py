import json
import os
from pathlib import Path

import anyio
import nltk
import pandas as pd
from datasets import load_dataset
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, TextEvals
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")

BUCKET_NAME = "ml_ops_project_g7"  # Used for saving predictions


def sentiment_to_numeric(sentiment: str) -> int:
    """Convert sentiment class to numeric."""
    if sentiment == "negative":
        return 0
    return 1  # positive


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="content"), DataDriftPreset(), DataSummaryPreset()])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save("text_overview_report.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data, class_names
    # Load Rotten Tomatoes dataset (train split) as reference data
    dataset = load_dataset("rotten_tomatoes", split="train")
    training_data = dataset.to_pandas()
    
    # Rename columns to match Evidently expectations and our internal naming
    # Rotten Tomatoes has 'text' and 'label'
    training_data = training_data.rename(columns={"text": "content", "label": "target"})
    
    class_names = ["negative", "positive"]

    yield

    del training_data, class_names


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("predictions/prediction_*.json")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    reviews, sentiment = [], []
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            reviews.append(data["review"])
            sentiment.append(sentiment_to_numeric(data["sentiment"]))
    dataframe = pd.DataFrame({"content": reviews, "sentiment": sentiment})
    dataframe["target"] = dataframe["sentiment"]
    return dataframe


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="predictions/prediction_")
    blobs = sorted(blobs, key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    for blob in latest_blobs:
        blob.download_to_filename(blob.name)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
