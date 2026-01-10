import torch
import torch.nn as nn
import math


def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]
    device = q.device

    # decide if we have heads or not (3D vs 4D input)
    has_heads = (q.dim() == 4)

    # Compute scaled dot-product scores: QK^T / sqrt(embed_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embed_dim)

    # Build the sliding window mask (band matrix).
    # True means "allowed attention", False means "blocked".
    half = window_size // 2
    # A neat trick: create a diagonal band using triangular matrices.
    window_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(-half).tril(half)

    # We want to mask OUTSIDE the window -> set them to a very negative number 
    # before softmax.
    neg_inf = torch.finfo(scores.dtype).min
    # expand mask to match scores dimensions
    if has_heads:
        # scores: [B, H, L, L]
        scores = scores.masked_fill(~window_mask.view(1, 1, seq_len, seq_len), neg_inf)
    else:
        # scores: [B, L, L]
        scores = scores.masked_fill(~window_mask.view(1, seq_len, seq_len), neg_inf)

    if padding_mask is not None:
        # padding_mask: 1 for real tokens, 0 for padding
        key_is_pad = (padding_mask == 0)
        if has_heads:
            # Broadcast over heads and query positions
            scores = scores.masked_fill(key_is_pad[:, None, None, :], neg_inf)
        else:
            scores = scores.masked_fill(key_is_pad[:, None, :], neg_inf)

    #softmax over keys dimension to get attention probabilities
    attention = torch.softmax(scores, dim=-1)

    # weighted sum of values
    values = torch.matmul(attention, v)

    # If queries are padding, force their outputs to be zero.
    if padding_mask is not None:
        query_is_real = (padding_mask != 0)
        if has_heads:
            attention = attention * query_is_real[:, None, :, None].to(attention.dtype)
            values = values * query_is_real[:, None, :, None].to(values.dtype)
        else:
            attention = attention * query_is_real[:, :, None].to(attention.dtype)
            values = values * query_is_real[:, :, None].to(values.dtype)
    
    return values, attention


class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # call the sliding window attention function you implemented
        # q,k,v are already in multi-head format: [B, H, L, head_dim]
        # In addition we forward the padding_mask so padded tokens won't be attended to.
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(
            q, k, v,
            window_size = self.window_size,
            padding_mask = padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''

        # ====== YOUR CODE: ======
        # attention output has same shape as x:
        attn_out = self.self_attn(x, padding_mask, return_attention=False)

        # residual connection then dropout then layer norm (Add & Norm)
        x = self.norm1(x + self.dropout(attn_out))

        # --- Feed-forward block ---
        ff_out = self.feed_forward(x)

        # residual connection then dropout then layer norm (Add & Norm)
        x = self.norm2(x + self.dropout(ff_out))
        # ========================
        
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # ====== YOUR CODE: ======
        # token embedding
        x = self.encoder_embedding(sentence) 
    
        # add positional encoding
        x = self.positional_encoding(x)      
    
        # dropout on embeddings (keep exactly here)
        x = self.dropout(x)
    
        # encoder stack
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
    
        # use CLS token (first token) for classification
        cls_vec = x[:, 0, :]                
    
        # classification head
        output = self.classification_mlp(cls_vec) 
        # ========================

        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    