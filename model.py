import torch
import torch.nn as nn
import tokenizer as Tokenizer
import data_loader as dl
import math

import torch.nn.functional as F

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
        x = x + self.pe[:, :x.shape[1]].detach()
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-5) -> None:
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
    
# class MultiHeadAttentionBlock(nn.Module):

#     def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.head_dim = embedding_dim // num_heads #called d_k in the paper

#         self.w_q = nn.Linear(embedding_dim, embedding_dim)
#         self.w_k = nn.Linear(embedding_dim, embedding_dim)
#         self.w_v = nn.Linear(embedding_dim, embedding_dim)

#         self.linear_out = nn.Linear(embedding_dim, embedding_dim) # W_o in the paper

#         self.dropout = nn.Dropout(dropout)

#     @staticmethod
#     def attention(query, key, value, dropout: nn.Dropout, mask=None):
#         d_k = query.size(-1)
#         scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

#         if mask is not None:
#             # Vérifie que mask est broadcastable
#             if mask.dim() == 2:
#                 mask = mask.unsqueeze(0).unsqueeze(0)
#             elif mask.dim() == 3:
#                 mask = mask.unsqueeze(1)
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         # Stabilisation numérique : soustraction du max
#         scores = scores - scores.max(dim=-1, keepdim=True).values
#         attn = scores.softmax(dim=-1)

#         if dropout is not None:
#             attn = dropout(attn)


#         output = attn @ value
#         return output, attn


#     def forward(self, v, k, q, mask=None):
#         query = self.w_q(q) #   (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)
#         key = self.w_k(k) #     (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)
#         value = self.w_v(v) #   (Batch, sequence_length, embedding_dim) --> (Batch, sequence_length, embedding_dim)

#         query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, sequence_length, head_dim)
#         key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2) #         (Batch, num_heads, sequence_length, head_dim)
#         value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, sequence_length, head_dim)

#         x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask)

#         # (Batch, num_heads, sequence_length, head_dim) --> (Batch, sequence_length, embedding_dim)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.embedding_dim)

#         # (Batch, sequence_length, embedding_dim)
#         return self.linear_out(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embedding_dim // num_heads  # d_k

        # Matrices de projection
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.linear_out = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.attention_score = None  # pour debug/visualisation

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout = None, mask=None):
        """
        query: (B, H, tgt_len, d_k)
        key:   (B, H, src_len, d_k)
        value: (B, H, src_len, d_k)
        mask:  (B, 1, tgt_len, src_len)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, tgt_len, src_len)

        # ✅ Gestion du masque
        if mask is not None:
            # Convertit en bool si nécessaire
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Broadcast si besoin
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)
            # Applique le masque
            scores = scores.masked_fill(~mask, float('-inf'))

        # ✅ Stabilisation numérique
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            attn = dropout(attn)

        # ✅ Produit d’attention
        output = torch.matmul(attn, value)  # (B, H, tgt_len, d_k)
        return output, attn

    def forward(self, query, key, value, mask=None):
        B, tgt_len, _ = query.size()

        # 1️⃣ Projette en têtes multiples
        query = self.w_q(query).view(B, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key   = self.w_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.w_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # query, key, value : (B, H, seq_len, head_dim)

        # 2️⃣ Attention multi-têtes
        x, self.attention_score = self.attention(query, key, value, self.dropout, mask)

        # 3️⃣ Recombine les têtes
        x = x.transpose(1, 2).contiguous().view(B, tgt_len, self.embedding_dim)
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
    
# class DecoderBlock(nn.Module):

#     def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) ->None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

#     def forward(self, x, enc_out, src_mask, tgt_mask): # src_mask is coming from encoder
#         x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
#         src_mask = src_mask if src_mask is not None else torch.ones(
#             (x.size(0), 1, 1, enc_out.size(1)), device=x.device
#         )
#         x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask))
#         x = self.residual_connection[2](x, self.feed_forward_block)
#         return x

class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1️⃣ Self-Attention (masque causal)
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )

        # 2️⃣ Cross-Attention (encoder–decoder)
        # Assure-toi que src_mask est broadcastable à [batch, num_heads, tgt_len, src_len]
        batch_size, tgt_len = x.size(0), x.size(1)
        src_len = enc_out.size(1)

        if src_mask is None:
            src_mask = torch.ones((batch_size, 1, tgt_len, src_len), device=x.device)
        elif src_mask.dim() == 4 and src_mask.size(-2) == 1:
            # élargit la dimension pour matcher tgt_len si nécessaire
            src_mask = src_mask.expand(batch_size, 1, tgt_len, src_len)

        x = self.residual_connection[1](
            x, lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask)
        )

        # 3️⃣ Feed Forward
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
        return self.proj(x)
    
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
    
def build_transformer(vocab_size: int, embedding_dim: int = 1024, src_seq_len: int = 256, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1, d_ff = 2048) -> Transformer:
    # Embedding layers
    src_embedding = InputEmbeddings(vocab_size, embedding_dim)
    tgt_embedding = InputEmbeddings(vocab_size, embedding_dim)

    # Positional encodings (only one required)
    pos_encoding = PositionalEncoding(embedding_dim, src_seq_len, dropout)

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

def get_model():
    return build_transformer(16384)

if __name__ == "__main__":
    transformer = build_transformer(16384)
    tokenizer = Tokenizer.Tokenizer(16384)
    dataset = dl.get_ds(tokenizer) #a byte list

    #Keep 90% for training
    # train_ds_size = int(0.9 * len(dataset))
    # val_ds_size = len(dataset) - train_ds_size
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [train_ds_size, val_ds_size])
    # print(train_ds.dataset , "\n\n\n" , val_ds.dataset)
