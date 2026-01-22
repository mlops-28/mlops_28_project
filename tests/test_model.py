from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import pytest
import torch

from artsy.model import ArtsyClassifier
from tests import _PATH_CONFIGS


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_model_batch_size(batch_size: int) -> None:
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    model = ArtsyClassifier(cfg)
    input = torch.randn(batch_size, 3, 128, 128)
    output = model(input)
    assert output.shape == (batch_size, 5), f"Shape of output not equal to [{batch_size}, 5], but {output.shape}"


# def test_model_input_shape():
#     model = NetWorkItBaby()
#     with pytest.raises(ValueError, "Expected input to a 4D tensor"):
#         model(torch.randn(1, 2, 3))
#     with pytest.raises(ValueError, "Expected each sample to have shape [1, 28, 28]"):
#         model(torch.randn(1, 1, 28, 29))


def test_model_label_mapping() -> None:
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    model = ArtsyClassifier(cfg)
    assert cfg.model.num_classes == len(
        model.label_map
    ), f"Number of classes ({cfg.model.num_classes}) not equal to number of mapped labels ({len(model.label_map)})"


def test_model_output_shape() -> None:
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    model = ArtsyClassifier(cfg)
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    assert output.shape == (
        1,
        cfg.model.num_classes,
    ), f"Shape of output not equal to [1, {cfg.model.num_classes}], but {output.shape}"
