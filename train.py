import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import (
    ProteinSequenceDataset,
    build_amino_acid_vocab,
    build_label_mapping,
    collate_protein_batch,
    encode_labels,
    load_protein_csv,
)
from src.model import ProteinClassifier
from src.utils import set_random_seed
from src.evalutation_utils import evaluate_model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    All arguments have defaults, so the script can be run with:

        python train.py
    """

    parser = argparse.ArgumentParser(
        description="Train an LSTM protein sequence classifier."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/proteins.csv",
        help="Path to the protein CSV dataset.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of examples in each batch.",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Size of each amino-acid embedding vector.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Size of the LSTM hidden state.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used by Adam.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of the dataset used for validation.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of the dataset used for testing.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_protein_lstm.pth",
        help="Path to save the best model.",
    )

    return parser.parse_args()


def choose_device() -> torch.device:
    """
    Select the best available training device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Check that command-line arguments contain valid values.
    """

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be greater than 0.")

    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be greater than 0.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")

    if args.validation_fraction <= 0:
        raise ValueError(
            "--validation-fraction must be greater than 0."
        )

    if args.test_fraction <= 0:
        raise ValueError(
            "--test-fraction must be greater than 0."
        )

    held_out_fraction = (
        args.validation_fraction
        + args.test_fraction
    )

    if held_out_fraction >= 1:
        raise ValueError(
            "The validation and test fractions must add "
            "to a value less than 1."
        )


def load_and_prepare_data(
    data_path: str,
) -> tuple[
    list[str],
    list[int],
    dict[str, int],
    dict[int, str],
    dict[str, int],
]:
    """
    Load raw protein sequences and prepare class labels.

    Protein sequences remain as strings here because
    ProteinSequenceDataset encodes them inside __getitem__().
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}"
        )

    sequences, labels = load_protein_csv(
        str(path)
    )

    if len(sequences) == 0:
        raise ValueError(
            "The dataset contains no protein sequences."
        )

    if len(sequences) != len(labels):
        raise ValueError(
            "The number of sequences does not match "
            "the number of labels."
        )

    vocab = build_amino_acid_vocab()

    label_to_index, index_to_label = build_label_mapping(
        labels
    )

    encoded_labels = encode_labels(
        labels,
        label_to_index,
    )

    return (
        sequences,
        encoded_labels,
        label_to_index,
        index_to_label,
        vocab,
    )


def split_data(
    sequences: list[str],
    labels: list[int],
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[int],
    list[int],
    list[int],
]:
    """
    Split raw protein sequences and encoded labels into
    training, validation, and test sets.
    """

    held_out_fraction = (
        validation_fraction
        + test_fraction
    )

    (
        train_sequences,
        held_out_sequences,
        train_labels,
        held_out_labels,
    ) = train_test_split(
        sequences,
        labels,
        test_size=held_out_fraction,
        random_state=random_seed,
        stratify=labels,
    )

    test_fraction_of_held_out = (
        test_fraction / held_out_fraction
    )

    (
        validation_sequences,
        test_sequences,
        validation_labels,
        test_labels,
    ) = train_test_split(
        held_out_sequences,
        held_out_labels,
        test_size=test_fraction_of_held_out,
        random_state=random_seed,
        stratify=held_out_labels,
    )

    return (
        train_sequences,
        validation_sequences,
        test_sequences,
        train_labels,
        validation_labels,
        test_labels,
    )


def create_datasets(
    train_sequences: list[str],
    validation_sequences: list[str],
    test_sequences: list[str],
    train_labels: list[int],
    validation_labels: list[int],
    test_labels: list[int],
    vocab: dict[str, int],
) -> tuple[
    ProteinSequenceDataset,
    ProteinSequenceDataset,
    ProteinSequenceDataset,
]:
    """
    Create PyTorch Dataset objects.

    Each dataset receives raw sequence strings and the vocabulary.
    The dataset encodes sequences when __getitem__() is called.
    """

    train_dataset = ProteinSequenceDataset(
        train_sequences,
        train_labels,
        vocab,
    )

    validation_dataset = ProteinSequenceDataset(
        validation_sequences,
        validation_labels,
        vocab,
    )

    test_dataset = ProteinSequenceDataset(
        test_sequences,
        test_labels,
        vocab,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def create_dataloaders(
    train_dataset: ProteinSequenceDataset,
    validation_dataset: ProteinSequenceDataset,
    test_dataset: ProteinSequenceDataset,
    batch_size: int,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create DataLoaders for training, validation, and testing.
    """

    collate_fn = collate_protein_batch

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    )


def create_model(
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
) -> ProteinClassifier:
    """
    Create the model and move it to the selected device.
    """

    model = ProteinClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        average_loss
        accuracy
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for sequences, sequence_lengths, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            sequences,
            sequence_lengths,
        )

        loss = loss_fn(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

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
            "The training DataLoader contains no examples."
        )

    average_loss = (
        total_loss / total_examples
    )

    accuracy = (
        total_correct / total_examples
    )

    return average_loss, accuracy


def print_training_configuration(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Print the experiment settings.
    """

    print(f"Using device: {device}")

    print("\nTraining configuration:")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(
        f"Validation fraction: "
        f"{args.validation_fraction}"
    )
    print(f"Test fraction: {args.test_fraction}")
    print(f"Random seed: {args.random_seed}")

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    label_to_index: dict[str, int],
    vocab: dict[str, int],
    embedding_dim: int,
    hidden_dim: int,
    model_path: str,
) -> None:
    path = Path(model_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "label_to_index": label_to_index,
        "vocab": vocab,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
    }

    torch.save(
        checkpoint,
        path,
    )
    
def main() -> None:
    args = parse_arguments()

    validate_arguments(args)

    set_random_seed(
        args.random_seed
    )

    device = choose_device()

    print_training_configuration(
        args,
        device,
    )

    # Load raw sequences and prepare labels.

    (
        sequences,
        encoded_labels,
        label_to_index,
        index_to_label,
        vocab,
    ) = load_and_prepare_data(
        args.data_path
    )

    print("\nDataset information:")
    print(f"Total examples: {len(sequences)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(
        f"Number of classes: "
        f"{len(label_to_index)}"
    )
    print(f"Label mapping: {label_to_index}")

    # Split the raw sequences and labels.

    (
        train_sequences,
        validation_sequences,
        test_sequences,
        train_labels,
        validation_labels,
        test_labels,
    ) = split_data(
        sequences=sequences,
        labels=encoded_labels,
        validation_fraction=(
            args.validation_fraction
        ),
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
   
    # Create Dataset objects.
   

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        test_sequences=test_sequences,
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
        vocab=vocab,
    )

    print("\nDataset split:")
    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )
    print(
        f"Test examples: "
        f"{len(test_dataset)}"
    )

   
    # Create DataLoaders.
    

    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    ) = create_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )

    # Create the model.

    model = create_model(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(label_to_index),
        device=device,
    )

    print("\nModel architecture:")
    print(model)

    # Create loss function and optimizer.

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Train.

    print("\nStarting training...\n")
    
    best_validation_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        
        validation_loss, validation_accuracy = evaluate_model(
            model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch + 1:02d}/"
            f"{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
            f" | Validation Loss: "
            f"{validation_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                validation_loss=validation_loss,
                label_to_index=label_to_index,
                vocab=vocab,
                model_path=args.model_path,
            )
            
            print(
                f"Best model saved at epoch "
                f"{epoch + 1:02d} with "
                f"validation loss: "
                f"{validation_loss:.4f}"
            )
            
    print("\nTraining complete.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    
    # These will be used when validation and testing
    # are added to the training pipeline.
    _ = validation_dataloader
    _ = test_dataloader
    _ = index_to_label


if __name__ == "__main__":
    main()