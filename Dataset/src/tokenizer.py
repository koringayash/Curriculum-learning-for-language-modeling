"""
2_tokenizer.py
Trains a Byte-Pair Encoding (BPE) tokenizer on the training corpus.
Used by reference model, scoring, and main training. Skips if exists.
"""
import os
import sys
import json
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import Dataset.config as cfg

def text_iterator(filepath: str):
    """Yields raw text strings from a JSONL file one at a time (memory efficient)."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]

def train_tokenizer():
    """Trains and saves BPE tokenizer. Skips if file exists."""
    import time
    start = time.time()
    if os.path.exists(cfg.TOKENIZER_FILE):
        print(f"✅ Tokenizer exists at {cfg.TOKENIZER_FILE}. Skipping.")
        return

    print("🤖 Training BPE tokenizer...")

    #Initialize tokenizer object
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    #When it train tokenizer it used one special character to indicate space so during training of tokenizer it split sentence into words and it add that special character before every word except first word of sentence so by making this Flag True it will do that explicitly.
    # "hello world"
    # ["hello", "Ġworld"] insted of this it will do ["Ġhello", "Ġworld"]

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.VOCAB_SIZE,
        min_frequency=5,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    
    os.makedirs(os.path.dirname(cfg.TOKENIZER_FILE), exist_ok=True)
    tokenizer.train_from_iterator(text_iterator(cfg.TRAIN_JSONL), trainer=trainer, length=(cfg.TRAIN_DOCS+2000))
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"✅ Tokenizer trained & saved in {time.time() - start:.1f}s (Vocab: {tokenizer.get_vocab_size():,})")