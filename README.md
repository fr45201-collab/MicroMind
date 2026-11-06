# MicroMind

![MicroMind Banner](MicroMind%20Neural%20Network%20Banner.png)

**A Small Language Model. Built From Scratch. Trained on a CPU.**

MicroMind is a ground-up implementation of a modern decoder-only Large Language Model (LLM), built entirely with **PyTorch** without relying on external AI frameworks such as Hugging Face Transformers.

The project aims to make modern language model architecture understandable through implementation rather than abstraction. Every major component—from tokenization and attention mechanisms to normalization, feed-forward networks, training, and text generation—is implemented as part of the project.

MicroMind demonstrates that building and training a functional language model does not necessarily require expensive GPU hardware.

It is a **6.88-million-parameter Transformer** trained from scratch on the TinyStories dataset using a standard laptop CPU.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Specifications](#key-specifications)
- [Architecture](#architecture)
- [Core Implementations](#core-implementations)
- [CPU Training Optimization](#cpu-training-optimization)
- [Dataset](#dataset)
- [Training Results](#training-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Text Generation](#text-generation)
- [Configuration](#configuration)
- [Technology Stack](#technology-stack)
- [Design Philosophy](#design-philosophy)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

Large Language Models are often treated as black boxes. Their internal architecture, mathematical operations, and training processes are hidden behind high-level libraries and pretrained model interfaces.

MicroMind takes a different approach.

Instead of loading an existing pretrained model, the project implements a decoder-only Transformer architecture and trains its parameters from random initialization.

The implementation incorporates several architectural improvements popularized by modern Transformer models, including:

- Rotary or positional attention components as configured by the model implementation.
- Root Mean Square Normalization (RMSNorm).
- SwiGLU feed-forward networks.
- Causal self-attention using PyTorch's scaled dot-product attention API.
- Autoregressive next-token prediction.

The result is a compact language model capable of generating short, coherent English stories.

## Key Specifications

| Specification | Details |
|---|---|
| Project | MicroMind |
| Architecture | Decoder-only Transformer |
| Parameters | 6.88 Million |
| Framework | PyTorch |
| Implementation | From scratch |
| Training hardware | CPU |
| Training dataset | TinyStories |
| Dataset size | 52.3 million tokens |
| Initial validation loss | 4.77 |
| Final validation loss | 0.7703 |
| Training duration | Approximately 8 hours |
| External AI frameworks | None |
| Primary language | Python |

## Architecture

MicroMind follows a modern LLaMA-inspired Transformer design rather than relying exclusively on the original GPT-1 architecture.

### 1. Decoder-Only Transformer

The model uses a decoder-only architecture designed for autoregressive language modeling.

Its primary objective is next-token prediction:

Given a sequence of tokens, the model predicts the probability distribution of the next token.

This process is repeated to generate text one token at a time.

### 2. RMSNorm

**Root Mean Square Normalization**

RMSNorm replaces conventional LayerNorm in the Transformer blocks.

It normalizes activations using their root mean square, reducing computational overhead while maintaining effective activation scaling.

### 3. SwiGLU

**Swish-Gated Linear Unit**

SwiGLU replaces the conventional GELU-based feed-forward network with a gated activation mechanism.

It combines two learned projections and a SiLU activation to provide a more expressive feed-forward transformation.

### 4. Scaled Dot-Product Attention

MicroMind implements causal self-attention using PyTorch's:

`torch.nn.functional.scaled_dot_product_attention`

This API provides optimized attention execution through supported PyTorch backends.

Causal masking ensures that each token can attend only to preceding tokens and itself, preserving autoregressive generation.

---

## Core Implementations

| Component | Purpose |
|---|---|
| Tokenizer | Converts text into numerical token representations |
| Embeddings | Maps token IDs into vector representations |
| Self-Attention | Captures contextual relationships between tokens |
| Causal Masking | Prevents access to future tokens |
| RMSNorm | Normalizes model activations |
| SwiGLU | Implements gated feed-forward transformations |
| Transformer Blocks | Combine attention and feed-forward operations |
| Language Modeling Head | Predicts next-token probabilities |
| Training Pipeline | Optimizes model parameters |
| Generation Pipeline | Produces autoregressive text |

All core model components are implemented using PyTorch.

---

## CPU Training Optimization

Training a Transformer on a CPU introduces significant computational and memory constraints.

MicroMind incorporates several techniques to make training practical on a laptop with limited resources.

### 1. Memory-Mapped Datasets

The implementation uses NumPy's `np.memmap` to access tokenized dataset data directly from binary files.

Instead of loading the entire dataset into memory, the training pipeline reads the required portions as needed.

Benefits:

- Reduced memory pressure.
- Efficient access to large tokenized datasets.
- Avoidance of unnecessary full-dataset copies.
- Better scalability for larger datasets.

### 2. Gradient Accumulation

MicroMind uses gradient accumulation to simulate a larger effective batch size.

| Parameter | Value |
|---|---:|
| Micro-batch size | 32 |
| Accumulation steps | 4 |
| Effective batch size | 128 |

The model accumulates gradients over four microbatches before performing an optimizer update.

This provides a larger effective batch without requiring the entire batch to fit into memory simultaneously.

### 3. Crash-Safe Checkpointing

Training checkpoints are periodically saved to preserve training progress.

The implementation saves checkpoints every 500 iterations.

This reduces the amount of lost work in case of unexpected interruptions.

For complete training resumption, checkpoints should preserve model weights, optimizer state, training step, and other relevant training state.

### 4. Resource-Conscious Training

The model is deliberately small, making it suitable for experimentation on resource-constrained hardware.

The implementation prioritizes:

- Controlled batch sizes.
- Efficient dataset access.
- Reduced memory consumption.
- Periodic checkpointing.
- CPU-compatible tensor operations.

---

## Dataset

MicroMind was trained on **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)**.

TinyStories is a synthetic storytelling dataset designed to study the language-learning capabilities of small language models.

### Dataset Specifications

| Metric | Value |
|---|---|
| Dataset | TinyStories |
| Reported token count | 52.3 million |
| Training objective | Next-token prediction |
| Data type | Text |
| Model initialization | Random initialization |

The dataset provides relatively simple English stories, making it suitable for experimenting with small-scale language models and autoregressive text generation.

---

## Training Results

MicroMind achieved a reported final validation loss of **0.7703**, starting from an initial validation loss of 4.77.

### Validation Loss

| Metric | Value |
|---|---:|
| Initial validation loss | 4.77 |
| Final validation loss | 0.7703 |
| Absolute reduction | 3.9997 |
| Relative reduction | Approximately 83.85% |

### Training Performance

- Approximately 8 hours of CPU training.
- 6.88 million trainable parameters.
- No pretrained model weights.
- No external Transformer framework.
- Training performed on standard laptop hardware.

### Text Generation

After training, MicroMind can generate short English stories involving characters and simple scenarios.

For example, prompts involving characters such as *Tom the robot* can produce short narrative continuations.

The model demonstrates learned language patterns, sentence structure, and punctuation.

*Note: Generated text quality varies depending on the prompt, sampling configuration, and model limitations.*

---

## Project Structure

The repository is organized into separate modules to keep model architecture, dataset processing, configuration, training, and generation independent.

```text
MicroMind/
│
├── checkpoints/
│   └── Training checkpoints
│
├── data/
│   └── Dataset and processed token data
│
├── config.py
│   └── Model and training configuration
│
├── data.py
│   └── Dataset preparation and data loading
│
├── tokenizer.py
│   └── Tokenization implementation
│
├── model.py
│   └── Transformer architecture
│
├── train.py
│   └── Training pipeline
│
├── generate.py
│   └── Autoregressive text generation
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│
├── .gitignore
│
└── MicroMind Neural Network Banner.png
```

---

## Installation

### Prerequisites

- Python 3.10 or compatible version.
- PyTorch.
- NumPy.
- Git.
- A CPU capable of running PyTorch workloads.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MicroMind.git
```

### 2. Navigate to the Project

```bash
cd MicroMind
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

MicroMind supports training from scratch through its dedicated training module.

Start the training pipeline:

```bash
python train.py
```

The training process initializes the model, loads the configured dataset, performs autoregressive language modeling, and periodically saves checkpoints.

**Important:** Configure the dataset paths, model hyperparameters, and training settings in `config.py` before starting training. Ensure the dataset has been prepared in the format expected by `data.py`.

---

## Text Generation

MicroMind includes a dedicated generation module for producing text using trained model checkpoints.

Run:

```bash
python generate.py
```

The generation script uses the trained model to generate autoregressive text.

The exact generation parameters, including prompt input, sampling temperature, and maximum generation length, depend on the implementation in `generate.py`.

---

## Configuration

Model and training settings are centralized in `config.py`.

This separation makes it easier to experiment with different model configurations without modifying the core Transformer implementation.

Configuration areas include:

- Model dimensions.
- Number of Transformer layers.
- Attention heads.
- Vocabulary configuration.
- Sequence length.
- Batch size.
- Gradient accumulation.
- Learning rate.
- Training iterations.
- Checkpoint frequency.

Refer to the actual configuration file for the current parameter values.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| PyTorch | Neural network implementation and training |
| NumPy | Numerical operations and memory-mapped data |
| TinyStories | Training dataset |
| Git | Version control |
| VS Code | Development environment |

MicroMind intentionally avoids external AI frameworks such as Hugging Face Transformers to keep the model implementation transparent and educational.

---

## Design Philosophy

MicroMind follows three fundamental principles:

**1. Understand Before Abstracting**

Implement the underlying mathematical operations rather than depending entirely on high-level abstractions.

**2. Optimize for Available Hardware**

Demonstrate how thoughtful memory management and training strategies can make small language models accessible without expensive GPU infrastructure.

**3. Build From First Principles**

Start with random model initialization and implement the complete training pipeline rather than relying on pretrained weights.

---

## Limitations

MicroMind is an experimental small language model, not a production-scale conversational AI system.

Its limitations include:

- Limited model capacity due to its 6.88-million-parameter architecture.
- Training data restricted to the TinyStories domain.
- Limited general knowledge and reasoning capabilities.
- Potential repetition and inconsistencies during generation.
- CPU training is substantially slower than GPU training for comparable workloads.
- No instruction tuning or alignment process is implied by the base training results.

The reported validation loss should not be interpreted as proof of general-purpose language understanding or zero overfitting.

---

## Future Improvements

Potential directions for further development include:

- Experimenting with different Transformer configurations.
- Improving tokenizer efficiency.
- Implementing additional evaluation metrics.
- Adding training and validation loss visualization.
- Improving generation strategies.
- Experimenting with larger datasets.
- Exploring model quantization.
- Benchmarking CPU and GPU inference.
- Investigating advanced attention optimizations.

---

## License

Specify the project's license here before distributing the repository.

---

## Acknowledgements

- [PyTorch](https://pytorch.org/) — Deep learning framework.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — Training dataset.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer architecture.
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Modern Transformer design reference.

---

**MicroMind — Understanding Language Models From the Inside Out.**

Built with PyTorch. Trained from scratch. Powered by a CPU.