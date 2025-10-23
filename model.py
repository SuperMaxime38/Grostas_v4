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

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, ff_dim) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embedding_dim) # W2 and b2

    def forward(self, x):
        # (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, ff_dim) --> (Batch, sequence_length, embedding_dim)
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embedding_dim // num_heads #called d_k in the paper

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.linear_out = nn.Linear(embedding_dim, embedding_dim) # W_o in the paper

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value,dropout: nn.Dropout, mask=None):
        d_k = query.shape[-1]

        # (Batch, num_heads, sequence_length, head_dim)
        attention_scores = (query @ key.transpose(-2, -1)) / d_k**0.5

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, v, k, q, mask=None):
        query = self.w_q(q) #   (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)
        key = self.w_k(k) #     (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)
        value = self.w_v(v) #   (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, sequence_length, head_dim)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2) #         (Batch, num_heads, sequence_length, head_dim)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, sequence_length, head_dim)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask)

        # (Batch, num_heads, sequence_length, head_dim) --> (Batch, sequence_length, embedding_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.embedding_dim)

        # (Batch, sequence_length, embedding_dim)
        return self.linear_out(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
        
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, tgt_mask): # src_mask is coming from encoder
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # (Batch, sequence_length, embedding_dim) -> (Batch, sequence_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, source_pos_encoding: PositionalEncoding, target_pos_encoding: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.source_pos_encoding = source_pos_encoding
        self.target_pos_encoding = target_pos_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.source_pos_encoding(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.target_pos_encoding(tgt)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(vocab_size: int, embedding_dim: int = 1024, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1, d_ff = 2048) -> Transformer:
    # Embedding layers
    src_embedding = InputEmbeddings(vocab_size, embedding_dim)
    tgt_embedding = InputEmbeddings(vocab_size, embedding_dim)

    # Positional encodings (only one required)
    pos_encoding = PositionalEncoding(embedding_dim)

    # Create the encoder blocks
    encoder_blocks = nn.ModuleList([EncoderBlock(
        MultiHeadAttentionBlock(embedding_dim, num_heads, dropout),
        FeedForwardBlock(embedding_dim, d_ff, dropout),
        dropout
    ) for _ in range(num_encoder_layers)])

    # Create the decoder blocks
    decoder_blocks = nn.ModuleList([DecoderBlock(
        MultiHeadAttentionBlock(embedding_dim, num_heads, dropout),
        MultiHeadAttentionBlock(embedding_dim, num_heads, dropout),
        FeedForwardBlock(embedding_dim, d_ff, dropout),
        dropout
    ) for _ in range(num_decoder_layers)])

    # Create the encoders and decoders
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    # Create the projection layer
    projection_layer = ProjectionLayer(embedding_dim, vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, pos_encoding, pos_encoding, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer