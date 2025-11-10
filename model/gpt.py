# 
# MLX GPT implementation based on Andrej Karpathy's GPT Tutorial.
#

import dataclasses
import model.transformer as transformer
import mlx.core as mx
import mlx.nn as nn
import numpy as np

@dataclasses.dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_heads: int
    n_embd: int
    n_layers: int
    dropout: float

class GPTLanguageModel(nn.Module):
    """ GPT Model"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # Transformer blocks
        self.blocks = nn.Sequential(*[transformer.Block(n_heads=config.n_heads, block_size=config.block_size, n_embd=config.n_embd, dropout=config.dropout) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd) # Final layer norm.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) # We want to output vocab_size logits at the end.

        # TODO: Add better initialization
        # self.apply_to_modules(self._init_weights)
    
    # TODO: Add better initialization
    def _init_weights(self, module):
        # if isinstance(module, nn.Linear):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #     if module.bias is not None:
        #         torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        pass

    def __call__(self, idx):
        B, T = idx.shape
        tok_embedding = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_embedding = self.positional_embedding_table(mx.arange(T)) # (T, n_embd)
        x = tok_embedding + pos_embedding # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        rng = np.random.default_rng()
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = nn.softmax(logits, axis=-1) # (B, vocab_size)
            # sample from the distribution
            probs = np.array(probs, copy=False) # (B, vocab_size)
            samples = rng.multinomial(n=100, pvals=probs) # (B, vocab_size)
            idx_next = mx.argmax(mx.array(samples), axis=-1, keepdims=True) # (B, 1)
            # idx_next = mx.argmax(probs, axis=-1, keepdims=True) # (B, 1)
            # append sampled index to the running sequence
            idx = mx.concat((idx, idx_next), axis=1) # (B, T+1)
        return idx
