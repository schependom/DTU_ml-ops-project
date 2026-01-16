import os

import pandas as pd


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
