from torch import nn 
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class ProteinClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.classifer = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim
        )
        
    def forward(
        self,
        sequences: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        
        embedded_sequences = self.embedding(sequences)
        
        packed_sequences = pack_padded_sequence(
            embedded_sequences,
            lengths=sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        lstm_output, (hidden_state, cell_state) = self.lstm(
            packed_sequences
        )
        
        final_hidden_state = hidden_state[-1]
        
        logits = self.classifer(final_hidden_state)
        
        return logits