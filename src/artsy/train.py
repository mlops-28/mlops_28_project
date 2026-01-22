import hydra
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

from artsy import _PATH_CONFIGS
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"

seed_everything(seed=42, workers=True)

log = logging.getLogger(__name__)


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml", version_base=None)
def train(cfg) -> None:
    print("Setting up...")
    dataset = WikiArtModule(cfg)
    dataset.setup()
    train_loader, val_loader = dataset.train_dataloader(), dataset.val_dataloader()
    model = ArtsyClassifier()

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # trainer = Trainer(accelerator=ACCELERATOR, callbacks=[checkpoint_callback]) # Check precision of input and model matches

    trainer = Trainer(
        accelerator=ACCELERATOR,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs if "trainer" in cfg and "max_epochs" in cfg.trainer else 999,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Done!")


if __name__ == "__main__":
    train()
