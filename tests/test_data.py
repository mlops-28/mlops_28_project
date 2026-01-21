from torch.utils.data import DataLoader
from hydra import compose, initialize
from omegaconf import DictConfig
from artsy.data import WikiArtModule
import pytest
import os
import torch


@pytest.mark.skipif(not os.path.exists("data/processed/"), reason="Data files not found")
def test_my_dataset():
    """Test the WikiArtModule class."""

    with initialize(config_path="configs", job_name="test"):
        cfg: DictConfig = compose(config_name="config")

    data = WikiArtModule(cfg)
    data.setup()

    image_size = cfg.data.hyperparameters.image_size
    nsamples = cfg.data.hyperparameters.nsamples
    labels_to_keep = cfg.data.hyperparameters.labels_to_keep
    nclasses = len(labels_to_keep)
    train_val_test = cfg.data.hyperparameters.train_val_test

    # Assert we have right total amount of data
    assert len(data.trainset) + len(data.valset) + len(data.testset) == int(nclasses * nsamples)

    # Assert that splits have been done properly
    assert len(data.testset) == int(nclasses * nsamples * train_val_test[2])
    assert len(data.valset) == int(nclasses * nsamples * train_val_test[1])
    assert len(data.trainset) == int(nclasses * nsamples * train_val_test[0])

    # Assert that images have the right shape
    image, _ = data.trainset[0]
    assert image.shape[-2:] == (image_size, image_size), f"Image shape is not {image_size}x{image_size}"

    # Assert that the images have the right dtype
    assert image.dtype == torch.float16

    trainloader = data.train_dataloader()
    testloader = data.test_dataloader()
    valloader = data.val_dataloader()

    # Assert that the data is imported in the right way
    assert isinstance(trainloader, DataLoader)
    assert isinstance(testloader, DataLoader)
    assert isinstance(valloader, DataLoader)

    # Assert that we only have the relevant labels
    def check_labels(data: DataLoader, type: str):
        invalid_labels = set()
        for _, labels in data:
            invalid_labels.update(set(labels.tolist()) - set(labels_to_keep))

        assert len(invalid_labels) == 0, f"There are styles included in {type}-dataset, which should have been included"

    check_labels(trainloader, "train")
    check_labels(testloader, "test")
    check_labels(valloader, "val")
