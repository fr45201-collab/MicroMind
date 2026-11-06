"""Data loading for MicroMind: download, tokenize, memory-mapped batch loading."""
import os
import urllib.request
import numpy as np
import torch

from tokenizer import CharTokenizer
from config import config

# TinyStories URL
DATA_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories-train.txt"
)
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # Back to 50 MB!


def download_data(path):
    """Download TinyStories if not already cached (capped at 50MB)."""
    if os.path.exists(path) and os.path.getsize(path) >= MAX_DOWNLOAD_BYTES:
        print(f"[data] found cached {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] downloading TinyStories (capped at 50MB) ...")
    
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "MicroMind/1.0"})
    downloaded = 0
    with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded >= MAX_DOWNLOAD_BYTES:
                break
    print(f"[data] downloaded {downloaded / 1024 / 1024:.1f} MB -> {path}")


def load_data():
    """Load text, build tokenizer, map to disk (saves RAM!)."""
    download_data(config["data_path"])

    with open(config["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print("[data] tokenizing... (this will take ~30 seconds)")

    # Convert to numpy array, then save as binary
    data_np = np.array(tokenizer.encode(text), dtype=np.uint16)
    n = int(0.9 * len(data_np))
    
    train_np = data_np[:n]
    val_np = data_np[n:]

    # Save to disk as .bin files
    train_np.tofile("data/train.bin")
    val_np.tofile("data/val.bin")

    print(f"[data] total tokens: {len(data_np):,}")
    print(f"[data] train tokens: {len(train_np):,}")
    print(f"[data] val tokens:   {len(val_np):,}")
    print(f"[data] vocab size:   {tokenizer.vocab_size}")

    # MEMORY MAP: This does NOT load into RAM. It reads from disk.
    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

    return tokenizer, train_data, val_data


def get_batch(split, train_data, val_data):
    """Sample a batch from the mmap."""
    data = train_data if split == "train" else val_data
    T = config["context_length"]
    B = config["batch_size"]

    max_start = len(data) - T - 1
    ix = torch.randint(0, max_start, (B,))

    # Read from the disk directly
    x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])

    return x, y