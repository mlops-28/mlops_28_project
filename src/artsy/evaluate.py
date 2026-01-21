import hydra
import logging
import os
from pytorch_lightning import Trainer
import torch

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml")
def evaluate(cfg) -> None:
    """Evaluating the trained art classification model"""
    print("Evaluating the trained model")

    dataset = WikiArtModule(cfg)
    dataset.setup()
    test_dataloader = dataset.test_dataloader()

    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(checkpoint_path=model_checkpoint, strict=True, map_location=DEVICE)

    trainer = Trainer(accelerator=ACCELERATOR, devices=1, logger=False, enable_checkpointing=False)

    results = trainer.test(model=model, dataloaders=test_dataloader, verbose=False)
    test_loss = results[0]["test_loss"]

    print("Test loss: ", test_loss)


if __name__ == "__main__":
    # typer.run(evaluate)
    print("Calling evaluate")
    evaluate()
