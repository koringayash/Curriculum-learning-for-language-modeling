"""
Evaluation/generator.py
Interactive side-by-side text generation from Curriculum vs Random models.

Usage:
    python Evaluation/generator.py

At each iteration you choose:
    [G] Generate  →  enter a prompt, see output from both models
    [E] Exit      →  quit
"""
import os
import sys
import torch

# ── Path setup (must happen before project imports) ───────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# These imports trigger sys.stdout redirect to logs.txt inside config files.
# We save the real stdout first and restore it immediately after.
_real_stdout = sys.__stdout__

import config_global
import Model.config as mc
from Evaluation.src.loader import find_latest_checkpoint, load_model, load_tokenizer

sys.stdout = _real_stdout          # restore console output for interactive use


# ── Generation ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """
    Autoregressive token-by-token generation.
      - temperature == 1.0  →  standard sampling
      - temperature  < 1.0  →  sharper / more confident
      - temperature  > 1.0  →  more random / creative
      - temperature == 0.0  →  greedy (argmax)
    """
    tokenizer.no_padding()
    tokenizer.no_truncation()

    ids = tokenizer.encode(prompt).ids
    if not ids:
        return "[empty prompt]"

    # Truncate prompt if it's already too long
    if len(ids) >= mc.SEQ_LEN:
        ids = ids[-(mc.SEQ_LEN - 1):]

    generated = list(ids)

    for _ in range(max_new_tokens):
        # Feed at most SEQ_LEN - 1 tokens as context
        ctx = generated[-mc.SEQ_LEN + 1:]
        x   = torch.tensor([ctx], dtype=torch.long, device=config_global.DEVICE)
        logits, _ = model(x)

        next_logits = logits[0, -1, :]         # (vocab_size,)

        if temperature == 0.0:
            next_token = next_logits.argmax().item()
        else:
            next_logits = next_logits / temperature
            probs       = torch.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

    # Decode only the newly generated tokens
    new_ids = generated[len(ids):]
    return tokenizer.decode(new_ids)


# ── Display ───────────────────────────────────────────────────────────────

def _print_side_by_side(prompt: str, curr_out: str, rand_out: str) -> None:
    W = 80
    print("\n" + "═" * W)
    print("  PROMPT".center(W))
    print("─" * W)
    print(f"  {prompt}")
    print("═" * W)
    print(f"  {'CURRICULUM MODEL':<38}  {'RANDOM MODEL'}")
    print("─" * W)

    # Split outputs into words and wrap at ~38 chars per column
    def wrap(text: str, width: int) -> list:
        lines, line = [], ""
        for word in text.split():
            if len(line) + len(word) + 1 <= width:
                line = (line + " " + word).lstrip()
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines or [""]

    c_lines = wrap(curr_out, 38)
    r_lines = wrap(rand_out, 38)
    n_rows  = max(len(c_lines), len(r_lines))

    for i in range(n_rows):
        c = c_lines[i] if i < len(c_lines) else ""
        r = r_lines[i] if i < len(r_lines) else ""
        print(f"  {c:<38}  {r}")

    print("═" * W + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  Model Generation Comparison — Curriculum vs Random")
    print("═" * 60)

    # Load checkpoints
    curr_ckpt = find_latest_checkpoint("curriculum")
    rand_ckpt = find_latest_checkpoint("random")

    if not curr_ckpt:
        print("❌ No curriculum checkpoint found. Run training first.")
        sys.exit(1)
    if not rand_ckpt:
        print("❌ No random checkpoint found. Run training first.")
        sys.exit(1)

    print(f"\n  Loading curriculum : {os.path.basename(curr_ckpt)}")
    curr_model = load_model(curr_ckpt)
    print(f"  Loading random     : {os.path.basename(rand_ckpt)}")
    rand_model = load_model(rand_ckpt)

    tokenizer = load_tokenizer()
    tokenizer.no_padding()
    tokenizer.no_truncation()

    print("\n  Both models loaded. Ready to generate.\n")

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        print("  Options:  [G] Generate   [E] Exit")
        choice = input("  > ").strip().lower()

        if choice in ("e", "exit", "q", "quit"):
            print("\n  Goodbye.\n")
            break

        if choice not in ("g", "generate"):
            print("  Please enter G or E.\n")
            continue

        # Get prompt
        prompt = input("\n  Enter prompt: ").strip()
        if not prompt:
            print("  Prompt cannot be empty.\n")
            continue

        # Get generation settings (with sensible defaults)
        try:
            max_tokens_input = input("  Max new tokens [default: 100]: ").strip()
            max_new_tokens   = int(max_tokens_input) if max_tokens_input else 100
            max_new_tokens   = max(1, min(max_new_tokens, mc.SEQ_LEN - 1))
        except ValueError:
            max_new_tokens = 100

        try:
            temp_input  = input("  Temperature [default: 0.8, 0=greedy]: ").strip()
            temperature = float(temp_input) if temp_input else 0.8
            temperature = max(0.0, temperature)
        except ValueError:
            temperature = 0.8

        print("\n  Generating", end="", flush=True)

        curr_out = generate(curr_model, tokenizer, prompt, max_new_tokens, temperature)
        print(".", end="", flush=True)

        rand_out = generate(rand_model, tokenizer, prompt, max_new_tokens, temperature)
        print(". done\n")

        _print_side_by_side(prompt, curr_out, rand_out)


if __name__ == "__main__":
    main()