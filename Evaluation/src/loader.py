"""
loader.py
Handles checkpoint discovery, model instantiation, and tokenizer loading.
Ensures both curriculum and random models exist before evaluation begins.
"""
import os
import re
import sys
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Model.config as mc
from Model.src.model import GPT

sys.stdout = open('logs.txt', 'a')

def find_latest_checkpoint(mode_tag: str) -> str | None:
    """
    Find the most recent checkpoint for a given training mode.
    Priority:
    1. Return *_final.pt if it exists
    2. Otherwise, return highest step checkpoint
    """
    ckpt_dir = mc.CHECKPOINT_DIR
    if not os.path.exists(ckpt_dir):
        return None

    files = os.listdir(ckpt_dir)

    # ✅ 1. Check for final checkpoint first
    final_name = f"{mode_tag}_final.pt"
    if final_name in files:
        return os.path.join(ckpt_dir, final_name)

    # ✅ 2. Otherwise search step-based checkpoints
    pattern = re.compile(
        rf"^{mode_tag}_step(\d+)\.pt$|ckpt_{mode_tag}_.*_step(\d+)\.pt$"
    )

    candidates = []

    for fname in files:
        match = pattern.search(fname)
        if match:
            step = int(match.group(1) or match.group(2))
            candidates.append((step, fname))

    if not candidates:
        return None

    # Return checkpoint with highest step
    candidates.sort(key=lambda x: x[0])
    return os.path.join(ckpt_dir, candidates[-1][1])

def load_model(checkpoint_path: str) -> GPT:
    """
    Load a trained GPT model from a checkpoint file.

    Args:
        checkpoint_path: Absolute path to the .pt checkpoint.

    Returns:
        GPT model in eval mode on the target device.
    """
    print(f"📦 Loading model from: {os.path.basename(checkpoint_path)}")
    ckpt = torch.load(checkpoint_path, map_location=config_global.DEVICE)

    model = GPT(
        vocab_size=mc.VOCAB_SIZE,
        n_embd=mc.N_EMBD,
        n_head=mc.N_HEAD,
        n_layer=mc.N_LAYER,
        seq_len=mc.SEQ_LEN,
        dropout=0.0  # Disable dropout for evaluation
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval().to(config_global.DEVICE)
    print("✅ Model loaded successfully.")
    return model


def load_tokenizer() -> Tokenizer:
    """
    Load the shared BPE tokenizer from disk.

    Returns:
        Trained Tokenizer instance.
    """
    tok_path = os.path.join(config_global.PROJECT_ROOT, "tokenizer", "bpe_tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"❌ Tokenizer not found at {tok_path}. Run Dataset/run.py first.")
    return Tokenizer.from_file(tok_path)