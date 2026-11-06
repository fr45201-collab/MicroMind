"""MicroMind configuration — single source of truth for all hyperparameters."""

config = {
    # ---- Model architecture ----
    "vocab_size": 106,     # TinyStories has 106 chars (was 65). Overwritten by train.py anyway.
    "context_length": 128, # Max tokens the model can attend to
    "n_embd": 192,         # Embedding dimension
    "n_head": 6,           # Attention heads
    "n_layer": 6,          # Transformer blocks
    "dropout": 0.1,        # Regularization

    # ---- Training ----
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_iters": 4000,     # More iters for bigger dataset (was 2500)
    "eval_interval": 250,
    "eval_iters": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    # ---- System ----
    "device": "cpu",
    "seed": 1337,

    # ---- Paths ----
    "data_path": "data/input.txt",
    "ckpt_path": "checkpoints/model.pt",
}

# Sanity check: embedding dim must split evenly across heads
assert config["n_embd"] % config["n_head"] == 0, \
    "n_embd must be divisible by n_head"