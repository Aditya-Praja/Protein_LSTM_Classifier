import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import (
    ProteinSequenceDataset,
    build_label_mapping,
    collate_protein_batch,
    encode_labels,
    load_protein_csv,
)
from src.evalutation_utils import (
    evaluate_model,
    load_checkpoint,
)
from src.model import ProteinClassifier
from src.utils import set_random_seed

from sklearn.metrics import classification_report, confusion_matrix


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained protein LSTM classifier."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/proteins.csv",
        help="Path to the protein CSV dataset.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_protein_lstm.pth",
        help="Path to the saved model checkpoint.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during evaluation.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Validation fraction used during training.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Test fraction used during training.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used to recreate the original split.",
    )

    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_raw_data(
    data_path: str,
) -> tuple[list[str], list[str]]:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}"
        )

    sequences, labels = load_protein_csv(
        str(path)
    )

    return sequences, labels


def recreate_test_split(
    sequences: list[str],
    encoded_labels: list[int],
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[str], list[int]]:
    """
    Recreate exactly the same test split used during training.
    """

    held_out_fraction = (
        validation_fraction
        + test_fraction
    )

    (
        _,
        held_out_sequences,
        _,
        held_out_labels,
    ) = train_test_split(
        sequences,
        encoded_labels,
        test_size=held_out_fraction,
        random_state=random_seed,
        stratify=encoded_labels,
    )

    test_fraction_of_held_out = (
        test_fraction / held_out_fraction
    )

    (
        _,
        test_sequences,
        _,
        test_labels,
    ) = train_test_split(
        held_out_sequences,
        held_out_labels,
        test_size=test_fraction_of_held_out,
        random_state=random_seed,
        stratify=held_out_labels,
    )

    return test_sequences, test_labels


def create_test_dataloader(
    test_sequences: list[str],
    test_labels: list[int],
    vocab: dict[str, int],
    batch_size: int,
) -> DataLoader:
    test_dataset = ProteinSequenceDataset(
        test_sequences,
        test_labels,
        vocab,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_protein_batch,
    )

    return test_dataloader


def create_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> ProteinClassifier:
    vocab = checkpoint["vocab"]
    label_to_index = checkpoint["label_to_index"]

    model = ProteinClassifier(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=len(label_to_index),
    )

    return model.to(device)

def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[list[int], list[int]]:
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for sequences, sequence_lengths, labels in dataloader:
            sequences = sequences.to(device)
            sequence_lengths = sequence_lengths.to(device)
            labels = labels.to(device)

            logits = model(sequences, sequence_lengths)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
    return all_predictions, all_true_labels

def main() -> None:
    args = parse_arguments()

    set_random_seed(
        args.random_seed
    )

    device = choose_device()

    print(f"Using device: {device}")

    checkpoint_path = Path(
        args.model_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: "
            f"{checkpoint_path.resolve()}"
        )

    # Load metadata before creating the model.
    checkpoint_metadata = torch.load(
        checkpoint_path,
        map_location=device,
    )

    vocab = checkpoint_metadata["vocab"]
    saved_label_to_index = checkpoint_metadata[
        "label_to_index"
    ]

    sequences, raw_labels = load_raw_data(
        args.data_path
    )

    current_label_to_index, _ = build_label_mapping(
        raw_labels
    )

    if current_label_to_index != saved_label_to_index:
        raise ValueError(
            "The dataset label mapping does not match "
            "the checkpoint label mapping."
        )

    encoded_labels = encode_labels(
        raw_labels,
        saved_label_to_index,
    )

    test_sequences, test_labels = recreate_test_split(
        sequences=sequences,
        encoded_labels=encoded_labels,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )

    test_dataloader = create_test_dataloader(
        test_sequences=test_sequences,
        test_labels=test_labels,
        vocab=vocab,
        batch_size=args.batch_size,
    )

    model = create_model_from_checkpoint(
        checkpoint=checkpoint_metadata,
        device=device,
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
    )

    print(
        f"Best model loaded from epoch "
        f"{checkpoint['epoch']:02d} "
        f"with validation loss "
        f"{checkpoint['validation_loss']:.4f}"
    )

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal evaluation on the test set:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    true_labels, predicted_labels = collect_predictions(
        model=model,
        dataloader=test_dataloader,
        device=device
    )
    
    class_names = [
        label 
        for label, index in sorted(
            saved_label_to_index.items(),
            key=lambda item: item[1]
        )
    ]
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
        )
    )
    
    confusion_mat = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names)))
    )
    
    print("\nConfusion Matrix:")
    print(confusion_mat)

if __name__ == "__main__":
    main()