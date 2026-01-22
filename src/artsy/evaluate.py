import hydra
import logging
import os
from pytorch_lightning import Trainer, seed_everything
import torch

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

seed_everything(seed=42, workers=True)

log = logging.getLogger(__name__)


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml", version_base=None)
def evaluate(cfg) -> None:
    """Evaluating the trained art classification model"""
    print("Evaluating the trained model")

    dataset = WikiArtModule(cfg)
    dataset.setup()
    test_dataloader = dataset.test_dataloader()

    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(
        checkpoint_path=model_checkpoint, cfg=cfg, strict=True, map_location=DEVICE
    )
    # Save_hyperparameters added to model, so cfg=cfg can be removed later

    model.eval()

    trainer = Trainer(accelerator=ACCELERATOR, devices=1, logger=False, enable_checkpointing=False)

    results = trainer.test(model=model, dataloaders=test_dataloader, verbose=False)
    test_loss, test_acc = results[0].values()

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {100 * test_acc:.2f}%")


if __name__ == "__main__":
    print("Running evaluate.py")
    evaluate()
