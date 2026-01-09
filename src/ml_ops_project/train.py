from pathlib import Path

from ml_ops_project.data import MyDataset
from ml_ops_project.model import Model


def train():
    dataset = MyDataset(Path("data/raw"))
    model = Model()
    # add rest of your training code here
    print(dataset, model)  # just to avoid unused variable warnings


if __name__ == "__main__":
    train()
