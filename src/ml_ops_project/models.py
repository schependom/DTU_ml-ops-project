import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification

from ml_ops_project.visualize import save_mismatches


class SentimentClassifier(pl.LightningModule):
    def __init__(self, model_name: str, inference_mode: bool, optimizer_cfg: DictConfig = None):
        super().__init__()
        self.optimizer_cfg = optimizer_cfg

        self.save_hyperparameters()

        if not inference_mode:  # Load the pre-trained model for binary classification (2 labels)
            if not optimizer_cfg:
                raise AttributeError("A 'NoneType' cannot be passed as optimizer config, when in training mode.")

            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            self.model.train()
            self.criterion = torch.nn.CrossEntropyLoss()

        else:
            # Fetch the saved model.
            path = "../models/epoch-epoch=01-val_accuracy=0.842.ckpt"
            self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
            self.model.eval()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # Log training accuracy and loss
        acc = self.train_acc(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # --- Save mismatches ---
        # Only doing this for the first few batches to avoid huge CSVs
        if batch_idx < 10:
            save_mismatches(batch, preds, folder="mismatches_val")
        # ---------------------------------------

        # Log validation accuracy and loss
        self.val_acc(preds, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", self.val_acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # --- Save mismatches for the test set ---
        save_mismatches(batch, preds, folder="mismatches_test")
        # -------------------------------------------------------

        # Log test accuracy and loss
        self.test_acc(preds, batch["labels"])
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
