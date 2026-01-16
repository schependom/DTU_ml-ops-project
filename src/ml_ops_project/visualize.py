import os

import pandas as pd
import torch
import wandb


def save_mismatches(batch, preds, folder="mismatches"):
    labels = batch["labels"].detach().cpu()
    preds = preds.detach().cpu()

    mask = preds != labels

    if mask.any():
        wrong_input_ids = batch["input_ids"][mask].detach().cpu()
        wrong_labels = labels[mask]
        wrong_preds = preds[mask]

        df = pd.DataFrame(
            {
                "input_ids": [ids.tolist() for ids in wrong_input_ids],
                "label": wrong_labels.tolist(),
                "predicted": wrong_preds.tolist(),
            }
        )

        os.makedirs(folder, exist_ok=True)
        df.to_csv(f"{folder}/errors.csv", mode="a", header=not os.path.exists(f"{folder}/errors.csv"), index=False)


def save_mismatches_to_wandb(batch, preds, tokenizer, table_name="mismatches"):
    # logs wrong predictions to a WandB table
    labels = batch["labels"].detach().cpu()
    preds = preds.detach().cpu()
    input_ids = batch["input_ids"].detach().cpu()
    
    mask = preds != labels

    if mask.any():
        # Initialize/Get the WandB Table
        columns = ["Input Text", "Label", "Predicted"]
        data = []

        # Extract and decode wrong instances
        wrong_input_ids = input_ids[mask]
        wrong_labels = labels[mask]
        wrong_preds = preds[mask]

        for i in range(len(wrong_labels)):
            text = tokenizer.decode(wrong_input_ids[i], skip_special_tokens=True)
            data.append([text, int(wrong_labels[i]), int(wrong_preds[i])])

        # Log the table to the current run
        wandb.log({table_name: wandb.Table(columns=columns, data=data)})
