import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class RottenTomatoesDataModule(pl.LightningDataModule):
    def __init__(self, model_name="distilbert-base-uncased", batch_size=32):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        # 1. Download the dataset
        dataset = load_dataset("rotten_tomatoes")

        # 2. Tokenize the data
        # We only tokenize here. Padding happens in the DataCollator (dynamic padding)
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        self.tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # 3. Set format for PyTorch
        self.tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # 4. Create the DataCollator (handles padding)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["train"],  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["validation"],  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["test"],  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )
