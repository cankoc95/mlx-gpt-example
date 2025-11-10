
import mlx.core as mx
import mlx.nn as nn

class Head(nn.Module):
    """ Single self attention head """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = mx.tril(mx.ones((block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        
        # compute attention scores ("affinities") based on scaled dot product attention from "Attention Is All You Need" paper.
        wei = q @ k.transpose(0, 2, 1) * k.shape[-1]**-0.5  # (B, T, T)
        # masked attention
        wei = mx.where(self.tril[:T, :T] != 0, wei, float('-inf')) # (B, T, T)
        wei = mx.softmax(wei, axis=-1) # (B, T, T)
        wei = self.dropout(wei) 

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) * (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multi head attention """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = [Head(head_size=head_size, n_embd=n_embd, block_size=block_size, dropout=dropout) for _ in range(num_heads)]
        # We concat each head of head_size num_heads times so we create a projection layer from head_size * num_heads.
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concat([head(x) for head in self.heads], axis=-1) # ()
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple feed forward layer """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    
    def __call__(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block"""
    def __init__(self, n_heads, block_size, n_embd, dropout):
        """
        n_heads: Total number of attention heads
        block_size: Block size or context length
        n_embd: Embedding dimension of the tokens, 
        dropout: Dropout
        """
        super().__init__()
        head_size = n_embd // n_heads # At the end, we want to get back n_embd so head_size should be n_embd // n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size, n_embd=n_embd, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
