from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from artsy.model import ArtsyClassifier
from artsy.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = ArtsyClassifier()

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model)

if __name__ == "__main__":
    train()
