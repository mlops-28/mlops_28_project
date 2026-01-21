import hydra
import logging
import os
from pytorch_lightning import Trainer
import torch

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"

log = logging.getLogger(__name__)

@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml")
# def evaluate(model_checkpoint: str = "models/model.pth"):
def evaluate(cfg) -> None:
    """Evaluating the trained art classification model"""
    print("Evaluating the trained model")
    
    dataset = WikiArtModule(cfg)
    dataset.setup()
    # breakpoint()
    test_dataloader = dataset.test_dataloader()

    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)

    model = ArtsyClassifier.load_from_checkpoint(checkpoint_path=model_checkpoint, strict=True, map_location=torch.device("cpu"))
    # breakpoint()

    trainer = Trainer(accelerator=ACCELERATOR, 
                      devices=1, 
                      logger=False, 
                      enable_checkpointing = False, 
                      precision="16-mixed",)

    results = trainer.test(model=model, dataloaders=test_dataloader, verbose=False)
    breakpoint()

    test_loss = results[0]["test_loss"]

    print("Test loss: ", test_loss)
    
    # # model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    # model.load_state_dict(torch.load(model_checkpoint)["state_dict"])
    
    

    # model.eval()
    # model.half()

    # correct, total = 0, 0

    # for images, labels in test_dataloader:
    #     y_pred = model(images)

    #     correct += (y_pred.argmax(dim=1) == labels).float().sum().item()

    #     total += labels.size(0)

    #     accuracy = correct / total

    #     print(f"The test accuracy is {accuracy}")

if __name__ == "__main__":
    # typer.run(evaluate)
    print("Calling evaluate")
    evaluate()