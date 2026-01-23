import pytest
import torch

from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.artsy.model import ArtsyClassifier
from src.artsy.data import WikiArtModule
from tests import _PATH_CONFIGS


processed_dir = Path("data/processed")
pt_files = list(processed_dir.glob("*.pt"))

def load_config() -> DictConfig:
    """Loads the config file for running the training script"""
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config")
    return cfg

def load_datamodule(cfg: DictConfig):
    datamodule = WikiArtModule(cfg)
    return datamodule.setup()

# def test_default_config() -> None:
#     """Tests that the config file is of type DictConfig, and containts the correct parameters"""
#     cfg = load_config()
#     assert isinstance(cfg, DictConfig), "cfg is not a DictConfig"

#     assert "trainer" in cfg, "trainer not in default_config"
#     assert "max_epochs" in cfg.trainer, "max_epochs not in training_conf"

#     assert "model" in cfg, "model not in default_config"
#     for k in ["in_channels", "num_classes", "lr", "label_map"]:
#         assert k in cfg.model, f"{k} not in model_conf"
#     assert "data" in cfg, "data not in default_config"
#     assert "hyperparameters" in cfg.data, "hyperparameters not in data_conf"
#     for param in ["seed", "batch_size", "image_size", "processed_data_path", "max_per_class",
#                   "nsamples", "labels_to_keep", "train_val_test"]:
#         assert param in cfg.data.hyperparameters, f"{param} not in hyperparameters"

@pytest.mark.skipif(len(pt_files) == 0, reason="Data files not found")
def test_datamodule_runs() -> None:
    """Tests the datamodule, when data is available"""
    cfg = load_config()
    breakpoint()
    datamodule = WikiArtModule(cfg)
    datamodule.setup()
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    assert train_loader is not None, "train_loader is None"
    assert val_loader is not None, "val_loader is None"

@pytest.mark.skipif(len(pt_files) > 0, reason="Data files found")
def test_datamodule_skips() -> None:
    """Smoke test of the datamodule, when data is not available"""
    cfg = load_config()
    breakpoint()
    datamodule = WikiArtModule(cfg)

    assert datamodule.train_dataloader() is not None, "Dataset module not have"
    # model = ArtsyClassifier(cfg)
    # model.eval()

    # batch_size = 2
    # x = torch.randn(batch_size, cfg.model.in_channels, 128, 128)

    # with torch.no_grad():
    #     y = model(x)

    # assert y.shape == (batch_size, cfg.model.num_classes), f"y does not have shape ({batch_size},{cfg.model.num_classes})"

    # assert torch.isfinite(y).all()
    # assert y.dtype == torch.float32

    # x = torch.randn(2, cfg.model.in_channels, 128, 128)

    # with torch.no_grad():
    #     y1 = model(x)
    #     y2 = model(x)

    # assert torch.equal(y1, y2)

    # trainer = Trainer(
    #     fast_dev_run=True,   
    #     accelerator="cpu",   
    # )

    # trainer.fit(model, datamodule)

    # checkpoint_dir = tmp_path / "models"

    # checkpoint_cb = ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     monitor="val_loss",
    #     save_top_k=1,
    # )

    # trainer = Trainer(
    #     accelerator="cpu",        
    #     fast_dev_run=True,         
    #     callbacks=[checkpoint_cb],
    #     enable_checkpointing=True,
    # )

    # trainer.fit(model, datamodule)

    # ckpts = list(checkpoint_dir.glob("*.ckpt"))
    # assert len(ckpts) > 0

    # assert checkpoint_cb.best_model_path != ""

    # trainer = Trainer(accelerator="cpu", fast_dev_run=True, logger=False, enable_checkpointing=False)
    # assert trainer.accelerator is not None
    # trainer.fit(model, datamodule)

    # assert "val_loss" in trainer.callback_metrics

    # accel = "mps" if torch.backends.mps.is_available() else "auto"
    # assert accel in {"mps", "auto"}

