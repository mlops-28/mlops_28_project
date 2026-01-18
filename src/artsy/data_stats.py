import torch
import glob
import typer
import matplotlib.pyplot as plt
from data import WikiArtModule
from utils import show_image_and_target
from hydra import compose, initialize
from omegaconf import DictConfig

def data_statistics(nimages: int = 15) -> None:
    """Loads WikiArtModule and computes class distribution and saves sample images"""
    with initialize(config_path="configs", job_name="test"):
        cfg: DictConfig = compose(config_name="config")

    data = WikiArtModule(cfg)
    data.setup()

    trainset = data.trainset
    testset = data.testset
    valset = data.valset

    print("Train dataset")
    print(f"Number of images: {len(trainset)}")
    print(f"Image shape: {trainset[0][0].shape}")
    print("\n")
    print(f"Test dataset")
    print(f"Number of images: {len(testset)}")
    print(f"Image shape: {testset[0][0].shape}")
    print(f"Val dataset")
    print(f"Number of images: {len(valset)}")
    print(f"Image shape: {valset[0][0].shape}")
    print()

    show_image_and_target(trainset.images[:nimages].to(torch.float32), trainset.target[:nimages], show=False)
    plt.savefig("reports/figures/samples_0_to_15.png")
    plt.close()

    train_label_distribution = torch.bincount(trainset.target)
    test_label_distribution = torch.bincount(testset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()

if __name__ == "__main__":
    typer.run(data_statistics)



