import torch
import typer
import matplotlib.pyplot as plt
from artsy.data import WikiArtModule
from types import SimpleNamespace


def data_statistics(nimages: int = 15) -> None:
    """Loads WikiArtModule and computes class distribution and saves sample images"""

    assert nimages <= 20

    # with initialize(config_path="configs", job_name="data_stats"):
    #     cfg: DictConfig = compose(config_name="config")

    cfg = SimpleNamespace(
        data=SimpleNamespace(
            hyperparameters=SimpleNamespace(
                seed=42,
                batch_size=32,
                image_size=128,
                processed_data_path="data/processed",
                nsamples=1000,
                labels_to_keep=[12, 21, 23, 9, 20],
                train_val_test=[0.8, 0.1, 0.1],
            )
        )
    )

    data = WikiArtModule(cfg)
    data.setup()

    trainset = data.trainset
    testset = data.testset
    valset = data.valset

    print("Train dataset")
    print(f"Number of images: {len(trainset)}")
    print(f"Image shape: {trainset[0][0].shape}")
    print("\n")
    print("Test dataset")
    print(f"Number of images: {len(testset)}")
    print(f"Image shape: {testset[0][0].shape}")
    print("\n")

    print("Val dataset")
    print(f"Number of images: {len(valset)}")
    print(f"Image shape: {valset[0][0].shape}")
    print()

    plt.figure()
    nrows = nimages // 5 + 1
    for i in range(nimages):
        plt.subplot(nrows, 5, i + 1)
        plt.imshow(trainset[i][0].to(torch.float32).permute(1, 2, 0).numpy())
        plt.title(f"Image {i+1} target = {int(trainset[i][1])}")
    plt.savefig("reports/figures/samples_0_to_15.png")
    plt.close()

    train_targets = torch.tensor([target for _, target in trainset])
    test_targets = torch.tensor([target for _, target in testset])

    unique_labels, counts = torch.unique(train_targets, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    unique_labels, counts = torch.unique(test_targets, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(data_statistics)
