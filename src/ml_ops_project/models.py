import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification


class SentimentClassifier(pl.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased", learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()  # Logs 'model_name' and 'learning_rate' automatically

        # Load the pre-trained model for binary classification (2 labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss

        # Calculate accuracy
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["labels"]).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss

        # Calculate accuracy
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["labels"]).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"])
