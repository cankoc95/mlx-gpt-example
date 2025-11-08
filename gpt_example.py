# 
# MLX GPT implementation based on Andrej Karpathy's GPT Tutorial.
#

import pathlib
import time
import dataclasses
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 384
dropout = 0.2
n_heads = 6
learning_rate = 3e-4
n_layers = 6

np.random.seed(1332)

data_dir = 'data/shakespeare'
if not pathlib.Path(data_dir + '/input.txt').exists():
    import subprocess

    try:
        # Let's download the tiny shakespeare dataset
        res = subprocess.run(['wget', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', '-P', data_dir], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Couldn't download dataset, exiting..")
        exit(1)

# read it in to inspect it
with open(pathlib.Path(data_dir + '/input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

class ShakespeareCharTokenizer:
    """Character level tokenizer for the tinyShakespeare dataset"""
    def __init__(self, text: list):
        self.unique_chars = set(sorted(list(text)))
        self.stoi = {c:i for i, c in enumerate(self.unique_chars)}
        self.itos = {i:c for i, c in enumerate(self.unique_chars)}

    def encode(self, s: str) -> mx.array:
        return mx.array([self.stoi[c] for c in s], dtype=mx.int64)

    def decode(self, indices: mx.array) -> str:
        return ''.join([self.itos[i] for i in indices])
    
    @property
    def vocab_size(self) -> int:
        return len(self.unique_chars)

tokenizer = ShakespeareCharTokenizer(text)
vocab_size = tokenizer.vocab_size

# let's now encode the entire text dataset and store it into an mlx array
data = mx.array(tokenizer.encode(text), dtype=mx.int64)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, block_size):
    """
    Generate a small batch of data of inputs x and targets y
    :param split: Training or validation split
    :param batch_size: Batch size to generate
    :param block_size: Block size (context or number of tokens) to generate in each batch.
    :returns: Generated inputs and targets of shape (batch_size, block_size)
    """
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))
    x = mx.stack([data[i.item():i.item() + block_size] for i in ix])
    y = mx.stack([data[(i+1).item():i.item() + block_size + 1] for i in ix])
    return x, y

class Head(nn.Module):
    """ Single self attention head """
    def __init__(self, head_size):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        # We concat each head of head_size num_heads times so we create a projection layer from head_size * num_heads.
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concat([head(x) for head in self.heads], axis=-1) # ()
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple feed forward layer """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    
    def __call__(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block"""
    def __init__(self, n_embd, n_heads):
        """n_embd: Embedding dimension of the tokens, n_heads: total number of heads we'd like"""
        super().__init__()
        head_size = n_embd // n_heads # At the end, we want to get back n_embd so head_size should be n_embd // n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@dataclasses.dataclass
class GPTConfig:
    n_embd: int
    dropout: float
    n_heads: int
    n_layers: int

class GPTLanguageModel(nn.Module):
    """ GPT Model"""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm.
        self.lm_head = nn.Linear(n_embd, vocab_size) # We want to output vocab_size logits at the end.

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
            idx_cond = idx[:, -block_size:]
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


def loss_fn(model, idx, targets):
    """
    Evaluate the model on given idx and targets and compute loss.
    :param idx: Inputs of shape (B, T)
    :param targets: Targets of shape (B, T)
    :returns: Cross entropy loss
    """
    logits = model(idx)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    return loss

def estimate_loss(model, loss_fn, batch_size, block_size):
    out = {}
    for split in ['train', 'val']:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size)
            loss, _ = loss_fn(model, X, Y)
            losses[k] = loss
        out[split] = mx.mean(losses).item()
    return out


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)

    model = GPTLanguageModel()
    optimizer = optim.AdamW(learning_rate=learning_rate)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print("Starting training")

    for iter in range(max_iters):
        tic = time.perf_counter()

        # Evaluate model on training & validation sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, loss_and_grad_fn, batch_size, block_size)
            # This will evaluate the graph.
            toc = time.perf_counter()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {toc-tic:.3f} s")

        # sample batch of data
        xb, yb = get_batch('train', batch_size, block_size)

        # get the loss and gradients
        loss, grads = loss_and_grad_fn(model, xb, yb)

        # Update optimizer state and model parameters
        optimizer.update(model, grads)

        # Force computation
        mx.eval(model.parameters(), optimizer.state)

    context = mx.zeros((1, 1), dtype=mx.int64)
    print(tokenizer.decode(model.generate(context, 500)[0].tolist()))

