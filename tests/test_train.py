from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from artsy.data import WikiArtModule
from artsy.model import ArtsyClassifier
from tests import _PATH_CONFIGS

processed_dir = Path("data/processed")
pt_files = list(processed_dir.glob("*.pt"))


def load_config() -> DictConfig:
    """Loads the config file for running the training script."""
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")
    return cfg


def test_default_config() -> None:
    """Tests that the config file is of type DictConfig, and containts the correct parameters."""
    cfg = load_config()
    assert isinstance(cfg, DictConfig), "cfg is not a DictConfig"

    assert "trainer" in cfg, "trainer not in default_config"
    assert "max_epochs" in cfg.trainer, "max_epochs not in training_conf"

    assert "model" in cfg, "model not in default_config"
    for k in ["in_channels", "num_classes", "lr", "label_map"]:
        assert k in cfg.model, f"{k} not in model_conf"
    assert "data" in cfg, "data not in default_config"
    for param in [
        "seed",
        "batch_size",
        "image_size",
        "processed_data_path",
        "max_per_class",
        "nsamples",
        "labels_to_keep",
        "train_val_test",
    ]:
        assert param in cfg.data, f"{param} not in data"


@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_datamodule_runs() -> None:
    """Tests the datamodule, when data is available."""
    cfg = load_config()
    datamodule = WikiArtModule(cfg)
    datamodule.setup()
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    assert train_loader is not None, "train_loader is None"
    assert val_loader is not None, "val_loader is None"


@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_trainer_cpu() -> None:
    """Tests that the trainer can be created on CPU."""
    trainer = Trainer(accelerator="cpu", max_epochs=1)
    assert trainer.accelerator is not None


@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_training_smoke() -> None:
    """Smoketest, to see that training starts and completes without crashing."""
    cfg = load_config()
    datamodule = WikiArtModule(cfg)
    datamodule.setup()

    model = ArtsyClassifier(cfg)

    trainer = Trainer(fast_dev_run=True, accelerator="cpu")

    trainer.fit(model, datamodule)


@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_checkpoints(tmp_path: Path) -> None:
    """Tests that that checkpoints are created when training the model."""
    cfg = load_config()
    datamodule = WikiArtModule(cfg)
    datamodule.setup()

    model = ArtsyClassifier(cfg)

    checkpoint_dir = tmp_path / "models"

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_top_k=1,
    )

    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        callbacks=[checkpoint_cb],
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule)

    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    assert len(ckpts) > 0, "Checkpoints are not created"

    assert checkpoint_cb.best_model_path != "", "Path is not created"


@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_loss_logging() -> None:
    """Tests that loss is logged and not negative"""
    cfg = load_config()
    datamodule = WikiArtModule(cfg)
    datamodule.setup()

    model = ArtsyClassifier(cfg)

    trainer = Trainer(accelerator="cpu", fast_dev_run=True, logger=False, enable_checkpointing=False)

    trainer.fit(model, datamodule)

    assert "train_loss" in trainer.callback_metrics, "Train loss is not in the callback metrics"
    assert "val_loss" in trainer.callback_metrics, "Validation loss is not in the callback metrics"

    train_loss = trainer.callback_metrics["train_loss"]
    val_loss = trainer.callback_metrics["val_loss"]

    assert torch.is_tensor(train_loss), "Train loss is not a tensor"
    assert torch.is_tensor(val_loss), "Validation loss is not a tensor"

    assert train_loss.item() >= 0.0, "Train loss is negative"
    assert val_loss.item() >= 0.0, "Validation loss is negative"
