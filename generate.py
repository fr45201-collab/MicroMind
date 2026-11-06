"""Generate text from a trained MicroMind checkpoint."""
import argparse
import torch

from config import config
from model import MicroMind
from tokenizer import CharTokenizer


def load_model(ckpt_path):
    """Rebuild model + tokenizer from a self-contained checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=config["device"])
    saved_config = ckpt["config"]

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = ckpt["stoi"]
    tokenizer.itos = ckpt["itos"]
    tokenizer.vocab_size = len(tokenizer.stoi)

    model = MicroMind(saved_config).to(config["device"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    return model, tokenizer


def generate_text(model, tokenizer, prompt, args):
    """Helper to encode, generate, and decode."""
    prompt_ids = []
    for ch in prompt:
        if ch in tokenizer.stoi:
            prompt_ids.append(tokenizer.stoi[ch])
            
    if not prompt_ids:
        prompt_ids = [tokenizer.stoi.get("\n", 0)]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=config["device"])

    out = model.generate(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    return tokenizer.decode(out[0].tolist())


def interactive_loop(model, tokenizer, args):
    """Run MicroMind in an interactive chat loop."""
    print("\n" + "=" * 60)
    print("🤖 Entering Interactive Mode. Type '/quit' to exit, '/reset' to clear.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if prompt.strip().lower() == "/quit":
            break
        if prompt.strip().lower() == "/reset":
            print("[Memory cleared. Start a new story.]")
            continue
        if not prompt.strip():
            continue

        print("\nMicroMind: ", end="")
        output = generate_text(model, tokenizer, prompt, args)
        print(output)
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=config["ckpt_path"])
    parser.add_argument("--interactive", action="store_true", help="Enter chat mode")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("[system] Loading model...")
    model, tokenizer = load_model(args.ckpt)

    if args.interactive:
        interactive_loop(model, tokenizer, args)
    else:
        # Standard one-shot generation
        print("=" * 60)
        print(f"PROMPT:  {args.prompt!r}")
        print(f"TEMP:    {args.temperature}")
        print(f"TOP_P:   {args.top_p}")
        print(f"REP_PEN: {args.repetition_penalty}")
        print("=" * 60)
        
        output = generate_text(model, tokenizer, args.prompt, args)
        print(output)
        print("=" * 60)


if __name__ == "__main__":
    main()