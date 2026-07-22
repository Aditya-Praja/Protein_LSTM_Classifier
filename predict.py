import argparse
from pathlib import Path

import torch

from src.model import ProteinClassifier

import argparse
from pathlib import Path

import torch

from src.model import ProteinClassifier


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Predict the class of a protein sequence "
            "using a trained LSTM classifier."
        )
    )

    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Protein amino-acid sequence to classify.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_protein_lstm.pth",
        help="Path to the trained model checkpoint.",
    )

    return parser.parse_args()


def choose_device() -> torch.device:
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clean_sequence(sequence: str) -> str:
   
    cleaned_sequence = "".join(sequence.split()).upper()
    
    return cleaned_sequence

def encode_sequence(
    sequence: str,
    vocab: dict[str, int],
) -> list[int]:
    
    unknown_token_index = vocab["<UNK>"]
    
    encoded_sequence = [
        vocab.get(amino_acid, unknown_token_index)
        for amino_acid in sequence
    ]
    
    return encoded_sequence

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> dict:
    
    path = Path(checkpoint_path)
    
    checkpoint = torch.load(path, map_location=device)
    
    return checkpoint

def create_model(
    checkpoint: dict,
    torch_device: torch.device
) -> ProteinClassifier:
    
    vocab = checkpoint["vocab"]
    label_to_index = checkpoint["label_to_index"]
    
    model = ProteinClassifier(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=len(label_to_index),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()
    
    return model

def predict_sequence(
    sequence: str,
    model: ProteinClassifier,
    vocab: dict[str, int],
    label_to_index: dict[int, str],
    device: torch.device
) -> tuple[str, float, dict[str, float]]:
    
    cleaned_seq = clean_sequence(sequence)
    encoded_seq = encode_sequence(cleaned_seq, vocab)
    sequence_tensor = torch.tensor([encoded_seq], dtype=torch.long, device=device)
    sequence_length = torch.tensor([len(encoded_seq)], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(sequence_tensor, sequence_length)
        
        probabilities = torch.softmax(logits, dim=1)
        
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_confidence = probabilities[0, predicted_index].item()
    
    index_to_label = {
        index: label
        for label, index in label_to_index.items()
    }
    
    predicted_label = index_to_label[predicted_index]
    class_probabilities = {
        index_to_label[idx] : prob.item()
        for idx, prob in enumerate(probabilities[0])
    }
    
    return predicted_label, predicted_confidence, class_probabilities

def print_prediction(
    sequence: str,
    predicted_label: str,
    predicted_confidence: float,
    class_probabilities: dict[str, float],
) -> None:

    print("\nPrediction results:")

    print(f"Sequence length: {len(sequence)}")

    print(
        f"Predicted class: "
        f"{predicted_label}"
    )

    print(
        f"Confidence: "
        f"{predicted_confidence:.4f}"
    )

    print("\nClass probabilities:")

    sorted_probabilities = sorted(
        class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    
    for label, probability in sorted_probabilities:
        print(
            f"{label:15s}: "
            f"{probability:.4f}"
        )
        
def main() -> None:
    args = parse_arguments()
    
    device = choose_device()
    
    checkpoint = load_checkpoint(args.model_path, device)
    
    model = create_model(checkpoint, device)
    
    vocab = checkpoint["vocab"]
    label_to_index = checkpoint["label_to_index"]
    
    predicted_label, predicted_confidence, class_probabilities = predict_sequence(
        sequence=args.sequence,
        model=model,
        vocab=vocab,
        label_to_index=label_to_index,
        device=device
    )
    
    print_prediction(
        sequence=args.sequence,
        predicted_label=predicted_label,
        predicted_confidence=predicted_confidence,
        class_probabilities=class_probabilities
    )
    
if __name__ == "__main__":
    main()
