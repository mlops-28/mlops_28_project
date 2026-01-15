from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

from artsy.model import ArtsyClassifier
from artsy.data import MyDataset

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"

def train():
    dataset = MyDataset("data/raw")
    model = ArtsyClassifier()

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    trainer = Trainer(accelerator=ACCELERATOR, callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model)

if __name__ == "__main__":
    train()
