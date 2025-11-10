import model.gpt as gpt

import pathlib
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

# Hyperparameters
block_size = 256
n_embd = 384
dropout = 0.2
n_heads = 6
n_layers = 6
dropout = 0.2
# Training
batch_size = 64
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4

np.random.seed(1392)

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

def get_batch(split):
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

def estimate_loss(model, loss_fn):
    out = {}
    for split in ['train', 'val']:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            loss, _ = loss_fn(model, X, Y)
            losses[k] = loss
        out[split] = mx.mean(losses).item()
    return out

def main(config: gpt.GPTConfig):
    mx.set_default_device(mx.gpu)
    
    model = gpt.GPTLanguageModel(config=config)
    optimizer = optim.AdamW(learning_rate=learning_rate)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print("Starting training")

    for iter in range(max_iters):
        tic = time.perf_counter()

        # Evaluate model on training & validation sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, loss_and_grad_fn)
            # This will evaluate the graph.
            toc = time.perf_counter()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {toc-tic:.3f} s")

        # sample batch of data
        xb, yb = get_batch('train')

        # get the loss and gradients
        loss, grads = loss_and_grad_fn(model, xb, yb)

        # Update optimizer state and model parameters
        optimizer.update(model, grads)

        # Force computation
        mx.eval(model.parameters(), optimizer.state)

    context = mx.zeros((1, 1), dtype=mx.int64)
    print(tokenizer.decode(model.generate(context, 500)[0].tolist()))


if __name__ == "__main__":
    config = gpt.GPTConfig(block_size=block_size, vocab_size=vocab_size, n_heads=n_heads, n_embd=n_embd, n_layers=n_layers, dropout=dropout)
    main(config)

