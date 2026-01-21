import os
import pytest
import torch

from src.artsy.model import ArtsyClassifier
# from tests import _PATH_DATA

def test_model_output_shape() -> None:
    model = ArtsyClassifier()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (1, 5), f"Shape of output not equal to [1, 5], but {output.shape}"

# def test_model_input_shape():
#     model = NetWorkItBaby()
#     with pytest.raises(ValueError, "Expected input to a 4D tensor"):
#         model(torch.randn(1, 2, 3))
#     with pytest.raises(ValueError, "Expected each sample to have shape [1, 28, 28]"):
#         model(torch.randn(1, 1, 28, 29))

@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_model_batch_size(batch_size: int) -> None:
    model = ArtsyClassifier()
    input = torch.randn(batch_size, 3, 256, 256)
    output = model(input)
    assert output.shape == (batch_size, 5), f"Shape of output not equal to [{batch_size}, 5], but {output.shape}"
