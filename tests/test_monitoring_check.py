import pandas as pd
from datasets import load_dataset
import sys
import os

sys.path.append(os.path.abspath("src"))

def test_dataset_loading():
    print("Testing dataset loading...")
    try:
        dataset = load_dataset("rotten_tomatoes", split="train")
        df = dataset.to_pandas()
        print(f"Dataset loaded. Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Verify renaming logic
        df = df.rename(columns={"text": "content", "label": "target"})
        print(f"Renamed columns: {df.columns}")
        
        assert "content" in df.columns
        assert "target" in df.columns
        print("Dataset loading test PASSED")
    except Exception as e:
        print(f"Dataset loading test FAILED: {e}")
        raise

def test_sentiment_logic():
    print("\nTesting sentiment logic...")
    def sentiment_to_numeric(sentiment: str) -> int:
        if sentiment == "negative":
            return 0
        return 1  # positive

    assert sentiment_to_numeric("negative") == 0
    assert sentiment_to_numeric("positive") == 1
    assert sentiment_to_numeric("foo") == 1 # Default case
    print("Sentiment logic test PASSED")

if __name__ == "__main__":
    test_dataset_loading()
    test_sentiment_logic()
