from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate a model without updating its parameters.

    Returns:
        average_loss
        accuracy
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for sequences, sequence_lengths, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(
                sequences,
                sequence_lengths,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    if total_examples == 0:
        raise ValueError(
            "The DataLoader contains no examples."
        )

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Load model weights and checkpoint metadata.
    """

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {path.resolve()}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint