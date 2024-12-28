
import torch
import torch.nn as nn
import math


### Start with Input embedding

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer size is (vocab_size, d_model)-->(6, 512)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)        # in paper we multiply with sqrt(d_model)
    
    
### Positional Encoding--> Positional encoding is added to give the model some information about the relative or absolute position of the tokens in the sequence. Vector of size 512

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:    # dropout to make model less overfit
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  
        self.dropout = nn.Dropout(dropout)
        
        # matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a position vector of shape (seq_len, 1) --> numerator of formula
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        # calculate div_term---> denominator of formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # 0::2 means start from 0 and jump 2 steps
        # apply the cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # 1::2 means start from 1 and jump 2 steps
        
        # add a batch dimension to positional encoding
        pe = pe.unsqueeze(0) # become tensor of (1, seq_len, d_model)--> added batch dimension as 1
        
        self.register_buffer('pe', pe) # register_buffer is used to add tensor to model's state_dict but not to be trained
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # add positional encoding to input or every token in the sequence, it will not change so requires_grad=False
        return self.dropout(x)
    
## Layer Normalization - Add and Norm part of encoder
class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable parameter alpha, multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # learnable parameter beta, added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # calculate mean along last dimension
        std = x.std(dim = -1, keepdim=True) # calculate std along last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # apply normalization and return
    
## Feed Forward layer - fully connected layer
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # linear layer 1, w1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Linear layer 2, w2 and B2
        
    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff)-->(Batch, Seq_len, d_model)
        # x = self.linear_1(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.linear_2(x)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
## Multihead Attention Block

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:  # h is number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # check if d_model is divisible by num_heads
        assert d_model % h == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // h
        
        # define matrices W_q, W_k, W_v , W_o
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv
        
        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    # calculate the attention
    def attention(query, key, value, d_k, mask=None, dropout=nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h, Seq_len, d_k) @ (Batch, h, d_k, Seq_len) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2, -1) // math.sqrt(d_k)) # calculate the attention scores
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) #  replace the attention scores with -1e9 where mask is 0
        attention_scores = torch.softmax(attention_scores, dim=-1) # (Batch, h, Seq_len, Seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask): # mask--> to mask the padding tokens, if we do not want some words to interact with each other
        query = self.w_q(q) # gives q prime --> (Batch, Seq_len, d_model)-> (Batch, Seq_len, d_model)
        key = self.w_k(k) # gives k prime --> (Batch, Seq_len, d_model)-> (Batch, Seq_len, d_model)
        value = self.w_v(v) # gives v prime --> (Batch, Seq_len, d_model)-> (Batch, Seq_len, d_model)
        
        # split the d_model into h heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, Seq_len, d_model)--> (Batch, Seq_len, h, d_k)-->(Batch, h, Seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # now calculate the attention, function above
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.d_k, mask, self.dropout)
        
        # concatenate the heads, (Batch, h , Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguos().view(x.shape[0], -1, self.h * self.d_k)
        
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        return self.w_o(x)
        
        
## skip connection and layer normalization in encoder - residual connection
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
## we have one big encoder block which has all the components of encoder, and it is in number Nx-- repeat the encoder block N times
## This block will have 2 Add and norm, one multihead attention and one feed forward block

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # first residual connection--> multihead attention (x, x, x)--> add and norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
## Encoder - N times encoder block- we can have N encoder blocks, lets define the encoder
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
    
    

        