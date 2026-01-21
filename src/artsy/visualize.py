import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from artsy import _PATH_CONFIGS, _PROJECT_ROOT
from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier

ACCELERATOR = "mps" if torch.backends.mps.is_available() else "auto"

log = logging.getLogger(__name__)

@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml")
def visualize(cfg) -> None:
    """Function to confusion matrix, and prediciton vs. target images"""
    print("Visualizing")
    dataset = WikiArtModule(cfg)
    dataset.setup(stage="test")
    test_dataloader = dataset.test_dataloader()

    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(
                                                checkpoint_path=model_checkpoint, 
                                                strict=True, 
                                                map_location=torch.device("cpu"),
                                                )
    
    model.eval()

    model.fc = torch.nn.Identity()

    embeddings_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in test_dataloader:
            images, targets = batch
            embeddings = model(images)
            embeddings_list.append(embeddings)
            targets_list.append(targets)
    
    embeddings = torch.cat(embeddings_list).cpu().numpy()
    targets = torch.cat(targets_list).cpu().numpy()

    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10,10))
    for label in sorted(set(targets)):
        mask = targets == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(label), s=5)
    
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{cfg.figure_name}")
    plt.close()

    # Plotting confusion matrix
    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    images_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, targets in test_dataloader:
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds_list.append(preds)
            images_list.append(images)
            targets_list.append(targets)
    
    preds = torch.cat(preds_list).cpu().numpy()
    images = torch.cat(images_list).cpu().numpy()
    targets = torch.cat(targets_list).cpu().numpy()

    inv_label_map = {v: k for k, v in model.label_map.items()}

    preds_orig = np.array([inv_label_map[p] for p in preds])
    targets_orig = np.array([inv_label_map[t] for t in targets])

    label_names = cfg.labels.names 

    pred_names = [label_names[l] for l in preds_orig]
    target_names = [label_names[l] for l in targets_orig]

    ordered_labels = sorted(label_names.keys())
    
    cm = confusion_matrix(targets_orig, preds_orig, labels=ordered_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_names[l] for l in ordered_labels])

    plt.figure(figsize=(8,8))
    disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{cfg.figures.confusion_matrix}")
    plt.close()

    # Plotting preidcted vs. true labels
    num_examples = 10
    indices = np.random.choice(len(images), num_examples, replaced = False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img = images[idx]

        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1,2,0))
        
        true_label = label_names[int(targets_orig[idx])]
        pred_label = label_names[int(preds_orig[idx])]

        ax.set_title(f"True: {true_label}. Predicted: {pred_label}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(cfg.figures.example_predictions)
    plt.close()