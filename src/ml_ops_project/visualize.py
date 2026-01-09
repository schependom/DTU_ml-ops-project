import pandas as pd
import os

def save_mismatches(batch, preds, folder="mismatches"):
    # Saves samples where the prediction does not match the label
    labels = batch["labels"]
    mask = preds != labels
    
    if mask.any():
        # Extract the wrong instances
        wrong_input_ids = batch["input_ids"][mask]
        wrong_labels = labels[mask]
        wrong_preds = preds[mask]

        # Note: In a real scenario, you'd decode input_ids back to text 
        # using a tokenizer, but for now we save the raw IDs/Results
        df = pd.DataFrame({
            "input_ids": [ids.tolist() for ids in wrong_input_ids],
            "label": wrong_labels.tolist(),
            "predicted": wrong_preds.tolist()
        })

        os.makedirs(folder, exist_ok=True)
        # Append to a file (or create new ones per batch)
        df.to_csv(f"{folder}/errors.csv", mode='a', header=not os.path.exists(f"{folder}/errors.csv"), index=False)