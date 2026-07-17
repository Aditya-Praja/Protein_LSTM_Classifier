from typing import List, Tuple, Dict

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'

def build_amino_acid_vocab() -> Dict[str, int]:
    
    vocab = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    
    for amino_acid in AMINO_ACIDS:
        vocab[amino_acid] = len(vocab)
        
    return vocab

def encode_sequence(
    sequence: str,
    vocab: Dict[str, int],
) -> List[int]:
    
    cleaned_sequence = sequence.strip().upper()
    
    unknown_index = vocab[UNKNOWN_TOKEN]
    
    encoded_sequence = [
        vocab.get(amino_acid, unknown_index)
        for amino_acid in cleaned_sequence
    ]
    
    return encoded_sequence

def load_protein_csv(
    csv_path: str,
) -> Tuple[List[str], List[str]]:
    
    df = pd.read_csv(csv_path)
    
    required_columns = {'sequence', 'label'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {required_columns}"
        )
        
    sequences = df['sequence'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()
    
    return sequences, labels

def build_label_mapping(
    labels: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    
    unique_labels = sorted(set(labels))
    
    label_to_index = {
        label: index
        for index, label in enumerate(unique_labels)
    }
    
    index_to_label = {
        index: label
        for label, index in label_to_index.items()
    }
    
    return label_to_index, index_to_label

def encode_labels(
    labels: List[str],
    label_to_index: Dict[str, int],
) -> List[int]:
    
    return [
        label_to_index[label]
        for label in labels
    ]
    
class ProteinSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        vocab: Dict[str, int],
    ) -> None:
        
        if len(sequences) != len(labels):
            raise ValueError(
                "The number of sequences must match the number of labels."
            )
            
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        encoded_sequence = encode_sequence(
            sequence=self.sequences[index],
            vocab=self.vocab
        )
        
        sequence_tensor = torch.tensor(
            encoded_sequence, 
            dtype=torch.long
        )
        
        label_tensor = torch.tensor(
            self.labels[index], 
            dtype=torch.long
        )
        
        return sequence_tensor, label_tensor
    
def collate_protein_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    sequences, labels = zip(*batch)
    
    sequence_lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )
    
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )
    
    labels = torch.stack(labels)

    return padded_sequences, sequence_lengths, labels
    