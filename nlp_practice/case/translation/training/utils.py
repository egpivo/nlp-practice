import torch

from nlp_practice.case.translation import PAD_TOKEN


def create_masks(
    input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor]:
    sequence_lengths = {"input": input.shape[1], "target": target.shape[1]}
    input_mask = torch.zeros(sequence_lengths["input"], sequence_lengths["input"]).type(
        torch.bool
    )
    init_target_mask = (
        (
            torch.triu(
                torch.ones(sequence_lengths["target"], sequence_lengths["target"])
            )
            == 1
        ).transpose(0, 1)
    ).float()
    target_mask = torch.ones_like(init_target_mask) * float("-inf")
    target_mask[target_mask == 1] = 0

    return input_mask.to(device), target_mask.to(device)


def create_padding_masks(
    input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor]:
    pad_sequence = lambda sequence: (sequence == PAD_TOKEN)
    return pad_sequence(input).to(device), pad_sequence(target).to(device)
