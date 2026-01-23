import logging
import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything, loggers
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
    """
    Train a CNN model for classifying art styles.

    Args:
        cfg: Config file containing parameters for training.

    """
    log.info("Setting up...")

    # Initialize data and model
    dataset = WikiArtModule(cfg)
    dataset.setup()
    train_loader, val_loader = dataset.train_dataloader(), dataset.val_dataloader()
    model = ArtsyClassifier(cfg)

    # Set up model directory depending on run location (cloud or locally)
    if os.path.exists("/gcs/wikiart-models"):
        model_dir = "/gcs/wikiart-models/models"
    else:
        model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=model_dir, monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")

    # Initialize trainer, syncing Hydra and WandB
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    trainer = Trainer(
        accelerator=ACCELERATOR,
        logger=loggers.WandbLogger(
            save_dir=f"{hydra_path}",
            project=f"{os.getenv("WANDB_PROJECT")}",
            log_model=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs if "trainer" in cfg and "max_epochs" in cfg.trainer else 999,
    )

    log.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    log.info("Finished training.")


if __name__ == "__main__":
    print("Running train.py")
    train()
