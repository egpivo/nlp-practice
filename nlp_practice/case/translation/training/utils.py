import torch

from nlp_practice.case.translation import PAD_TOKEN


def create_masks(
    input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor]:
    sequence_lengths = {"input": input.shape[0], "target": target.shape[0]}
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
    target_mask = init_target_mask.masked_fill(
        init_target_mask == 0, float("-inf")
    ).masked_fill(init_target_mask == 1, float(0))

    return input_mask.to_device(device), target_mask.to_device(device)


def create_padding_masks(
    input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor]:
    pad_sequence = lambda sequence: (sequence == PAD_TOKEN).transpose(0, 1)
    return pad_sequence(input).to_device(device), pad_sequence(target).to_device(device)
