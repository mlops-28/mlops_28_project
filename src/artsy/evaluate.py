import logging
import os

from dotenv import load_dotenv
import hydra
from pytorch_lightning import Trainer, seed_everything
import torch
import wandb

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
seed_everything(seed=42, workers=True)
log = logging.getLogger(__name__)
load_dotenv()


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml", version_base=None)
def evaluate(cfg) -> None:
    """Evaluating the trained art classification model"""
    log.info("Evaluating the trained model.")

    # Initialize data
    dataset = WikiArtModule(cfg)
    dataset.setup()
    test_dataloader = dataset.test_dataloader()

    log.info("Downloading model from WandB")
    api = wandb.Api()
    artifact_name = f"{os.getenv('WANDB_ENTITY')}-org/{cfg.eval.model_registry}/{cfg.eval.collection}:{cfg.eval.tag}"
    artifact = api.artifact(name=artifact_name)
    artifact.download(f"{cfg.registry.artifact_dir}")

    # Load model from downloaded checkpoint
    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(
        checkpoint_path=model_checkpoint, cfg=cfg, strict=True, map_location=DEVICE, weights_only=False
    )  # Save_hyperparameters added to model, so cfg=cfg can be removed later
    model.eval()

    # Initialize trainer and test model
    trainer = Trainer(accelerator=ACCELERATOR, devices=1, logger=False, enable_checkpointing=False)
    results = trainer.test(model=model, dataloaders=test_dataloader, verbose=False)
    test_loss, test_acc = results[0].values()

    log.info(f"Test loss: {test_loss:.4f}")
    log.info(f"Test accuracy: {100 * test_acc:.2f}%")


if __name__ == "__main__":
    print("Running evaluate.py")
    evaluate()
