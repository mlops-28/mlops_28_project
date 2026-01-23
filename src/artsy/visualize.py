import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"
log = logging.getLogger(__name__)


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml", version_base=None)
def visualize(cfg) -> None:
    """Function to plot losses, confusion matrix, and prediciton vs. target images"""
    log.info("Loading dataset, model, and checkpoints")
    dataset = WikiArtModule(cfg)
    dataset.setup(stage="test")
    test_dataloader = dataset.test_dataloader()

    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(
        checkpoint_path=model_checkpoint, cfg=cfg, strict=True, map_location=torch.device("cpu"), weights_only=False
    )

    model.eval()

    ### Plotting confusion matrix
    log.info("Running the trained model in inference mode")
    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    images_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, targets in test_dataloader:
            images = images.float()
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds_list.append(preds)
            images_list.append(images)
            targets_list.append(targets)

    preds = torch.cat(preds_list).cpu().numpy()
    images = torch.cat(images_list).cpu().numpy()
    targets = torch.cat(targets_list).cpu().numpy()
    inv_label_map = {v: k for k, v in model.label_map.items()}

    preds_orig = [inv_label_map[p.item()] for p in preds]
    preds_orig = torch.Tensor(preds_orig)

    label_names_df = cfg.visualize.labels.names
    label_names = list(label_names_df.values())

    pred_names = [label_names_df[label] for label in preds_orig.tolist()]
    target_names = [label_names_df[label] for label in targets.tolist()]

    log.info("Creating confusion matrix")
    cm = confusion_matrix(target_names, pred_names, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.figure(figsize=(8, 8))
    disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(f"./reports/figures/{cfg.visualize.figures.confusion_matrix}")
    plt.close()

    # Plotting preidcted vs. true labels
    log.info("Plotting true vs. predicted labels")
    num_examples = 10
    indices = np.random.choice(len(images), num_examples)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img = images[idx]

        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.transpose(1, 2, 0))

        true_label = label_names_df[int(targets[idx])]
        pred_label = label_names_df[int(preds_orig[idx])]

        ax.set_title(f"True: {true_label}\nPredicted: {pred_label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"./reports/figures/{cfg.visualize.figures.example_predictions}")
    plt.close()

    log.info("Done plotting.")


if __name__ == "__main__":
    print("Running visualize.py")
    visualize()
