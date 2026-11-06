"""Train MicroMind on Tiny Shakespeare."""
import os
import time
import math

import torch

# --- CPU speedup: use available cores ---
torch.set_num_threads(min(8, os.cpu_count() or 1))

from config import config
from data import load_data, get_batch
from model import MicroMind


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Average loss over eval_iters batches for both splits."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """Linear warmup then cosine decay to 10% of max LR."""
    warmup_iters = 100
    if it < warmup_iters:
        return config["learning_rate"] * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / (config["max_iters"] - warmup_iters)
    return config["learning_rate"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    # ---- Reproducibility ----
    torch.manual_seed(config["seed"])

    # ---- Data ----
    tokenizer, train_data, val_data = load_data()

    # Patch vocab_size into the config (overrides the placeholder)
    config["vocab_size"] = tokenizer.vocab_size

    # ---- Model ----
    model = MicroMind(config).to(config["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # ---- Training loop ----
    print(f"[train] training for {config['max_iters']} iterations on {config['device']}")
    print(f"[train] effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    t0 = time.time()
    accum_steps = config["gradient_accumulation_steps"]

    for it in range(config["max_iters"]):

        # Update LR (warmup + cosine decay)
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation
        if it % config["eval_interval"] == 0 or it == config["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            print(
                f"  iter {it:5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"{elapsed:6.1f}s"
            )

            # Save intermediate checkpoint every 500 iterations (except iter 0)
            if it > 0 and it % 500 == 0:
                ckpt_path = f"checkpoints/model_iter_{it}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                }, ckpt_path)
                print(f"  --> [saved checkpoint: {ckpt_path}]")

        # --- GRADIENT ACCUMULATION START ---
        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(accum_steps):
            x, y = get_batch("train", train_data, val_data)
            _, loss = model(x, y)
            # Scale loss to average out the accumulation
            loss = loss / accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        # --- GRADIENT ACCUMULATION END ---

    # ---- Save final checkpoint ----
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        config["ckpt_path"],
    )
    print(f"[train] saved final checkpoint to {config['ckpt_path']}")


if __name__ == "__main__":
    main()