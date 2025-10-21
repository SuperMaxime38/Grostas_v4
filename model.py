import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x) * self.embedding_dim ** 0.5
    
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim:int, seq_lenght: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_lenght = seq_lenght
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of (seq_lenght, embedding_dim)
        pe = torch.zeros(seq_lenght, embedding_dim)

        # Create a vector of shape (seq_lenght)
        position = torch.arange(0, seq_lenght, dtype=torch.float).unsqueeze(1) # (seq_lenght, 1)

        # Create a vector of shape (1, embedding_dim)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))

        # Apply the equation to each position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0) # (1, seq_lenght, embedding_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1]]).requires_grad(False)
        return self.dropout(x)
