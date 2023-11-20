import pytest
import torch

from nlp_practice.case.translation import PAD_TOKEN
from nlp_practice.case.translation.training.utils import (
    create_masks,
    create_padding_masks,
)


@pytest.fixture
def input_target_tensors():
    batch_size = 64
    input_size = 8
    output_size = 10
    input_tensor = torch.randint(
        0,
        10,
        (
            batch_size,
            input_size,
        ),
    )
    target_tensor = torch.randint(0, 10, (batch_size, output_size))
    return input_tensor, target_tensor


def test_create_masks(input_target_tensors):
    input_tensor, target_tensor = input_target_tensors

    input_mask, target_mask = create_masks(input_tensor, target_tensor)

    # Check if the masks have the correct shapes
    assert input_mask.shape == (8, 8)
    assert target_mask.shape == (10, 10)

    # Check if the input mask has the correct values
    assert torch.all(input_mask == 0)

    # Check if the input mask has the correct values
    assert torch.all(input_mask == torch.zeros_like(input_mask))


def test_create_padding_masks(input_target_tensors):
    # Mock data
    input_tensor, target_tensor = input_target_tensors

    # Create padding masks
    input_padding_mask, target_padding_mask = create_padding_masks(
        input_tensor, target_tensor
    )
    assert input_padding_mask.shape == (64, 8)
    assert torch.all(input_padding_mask == (input_tensor == PAD_TOKEN))
    assert target_padding_mask.shape == (64, 10)
    assert torch.all(target_padding_mask == (target_tensor == PAD_TOKEN))
