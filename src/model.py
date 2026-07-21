from torch import nn 
import torch

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
        
    def forward(
        self,
        sequences: torch.Tensor
    ) -> torch.Tensor:
        
        embedded_sequences = self.embedding(sequences)
        
        lstm_output, (hidden_state, cell_state) = self.lstm(
            embedded_sequences
        )
        
        return hidden_state