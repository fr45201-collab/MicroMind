# 🧠 MicroMind

![MicroMind Banner](MicroMind%20Neural%20Network%20Banner.png)

*A 6.88M parameter LLaMA-style GPT built completely from scratch in PyTorch.*

MicroMind is a character-level, decoder-only Transformer built from absolute zero. It was trained entirely on a laptop CPU (no GPU required) using the TinyStories dataset.

It features modern LLM architecture upgrades, including SwiGLU, RMSNorm, and PyTorch's fused SDPA.

## 📊 Model Specs

| Specification             | Details                                 |
| ------------------------- | --------------------------------------- |
| **Architecture**          | LLaMA-style Decoder-only Transformer    |
| **Parameters**            | 6,880,000 (6.88M)                       |
| **Layers**                | 8 (with RMSNorm)                        |
| **Attention Heads**       | 8                                       |
| **Embedding Dimension**   | 256                                     |
| **Context Length**        | 128 tokens                              |
| **Vocabulary Size**       | 106 (Character-level)                   |
| **Dataset**               | TinyStories (50MB subset, 52.3M tokens) |
| **Final Validation Loss** | 0.7703                                  |
| **Training Time**         | \~7.8 hours on a laptop CPU             |

## 🛠️ Features Implemented

### Modern Architecture

- LLaMA-style decoder-only Transformer.
- SwiGLU activation instead of standard GELU.
- RMSNorm instead of standard LayerNorm.
- PyTorch's fused Scaled Dot-Product Attention (SDPA).

### Training Optimizations

- **Optimized Attention:** Uses `F.scaled_dot_product_attention` for efficient attention computation.
- **Memory-Mapped Data:** Uses `np.memmap` to train on a 50MB dataset without loading the entire dataset into RAM.
- **Gradient Accumulation:** Simulates a batch size of 128 on a CPU by accumulating gradients over 4 steps.
- **Cosine Learning Rate Schedule:** Linear warmup followed by cosine decay.
- **Checkpointing:** Automatically saves intermediate checkpoints every 500 iterations to protect against training interruptions.

### Advanced Text Generation

- Repetition Penalty.
- Top-K Sampling.
- Top-P (Nucleus) Sampling.
- Temperature-based sampling.

These techniques help improve generation quality and reduce repetitive output.

## 🚀 Quick Start

### 1. Clone the Repository
```
git clone <your-repository-url>
cd MicroMind
```

### 2. Install Dependencies
```
pip install torch numpy
```

### 3. Train from Scratch
```
python train.py
```

This automatically downloads the 50MB dataset, tokenizes it, and begins the training process.

**Estimated training time:** \~8 hours on a laptop CPU.

### 4. Interactive Mode
```
python generate.py --interactive
```

Start an interactive session and chat with MicroMind.

### 5. Generate a Single Response
```
python generate.py --prompt "Once upon a time" --max_new_tokens 300 --temperature 0.8 --top_p 0.9 --repetition_penalty 1.3
```

## 📖 Sample Output

**Prompt:**
```
Once upon a time, there was a little robot named
```

**Generated Output:**

> Once upon a time, there was a little robot named Tom. Tom loved to play outside in the garden. He would see a tear from the sky.
>
> One day, the driver was not very hungry. It was his favorite car and wanted to buy a new toy. But soon other box and started to feel the squirrel. A big ball Max asked Jack, "Th...

## 🏆 Why MicroMind?

MicroMind demonstrates that you do not need a $2,000 GPU to understand and build modern AI architectures.

By implementing a Transformer architecture from scratch and optimizing it for CPU constraints, it is possible to train a functional language model on a standard laptop.

This project explores several important concepts in modern Large Language Model development:

- Transformer architecture and self-attention.
- Decoder-only language modeling.
- Tokenization and vocabulary construction.
- RMSNorm and SwiGLU.
- Scaled Dot-Product Attention.
- Gradient accumulation.
- Learning rate scheduling.
- Memory-efficient dataset processing.
- Text generation and sampling strategies.
- Model checkpointing and inference.

**MicroMind is a hands-on exploration of how modern language models work, from architecture implementation to training and text generation.**

---

*Built with Python and PyTorch. Trained on CPU. Built from scratch.*