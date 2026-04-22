"""
3_encoder.py
Encodes the full training corpus into a single flat uint16 binary file.
Also encodes the held-out test corpus into test_tokens.bin.
Saves document boundaries for fast stage-splitting later without re-tokenizing.
"""
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import Dataset.config as cfg

sys.stdout = open('logs.txt', 'a')

def _encode_file(jsonl_path: str, bin_path: str, desc: str) -> int:
    """
    Shared helper: tokenises every document in a JSONL file and writes
    the token IDs (uint16, with <bos>/<eos> wrapping) to a flat binary.
    Returns the total number of tokens written.
    """
    import time
    tokenizer = Tokenizer.from_file(cfg.TOKENIZER_FILE)
    tokenizer.no_padding()
    tokenizer.no_truncation()

    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    total_tokens = 0

    with open(bin_path, "wb", buffering=1024 * 1024) as f_out, \
         open(jsonl_path, "r", encoding="utf-8") as f_in:

        for line in tqdm(f_in, desc=desc, unit="doc"):
            text = json.loads(line)["text"]
            ids  = tokenizer.encode(text).ids
            buffer = np.array([bos] + ids + [eos], dtype=np.uint16)
            buffer.tofile(f_out)
            total_tokens += len(buffer)

    return total_tokens


def encode_corpus():
    """
    Tokenises train.jsonl  → train_tokens.bin  + doc_offsets.json
    Tokenises test.jsonl   → test_tokens.bin                        ← FIX
    Skips each output file if it already exists.
    """
    import time

    # ── Training corpus ────────────────────────────────────────────────
    if os.path.exists(cfg.TRAIN_TOKENS) and os.path.exists(cfg.DOC_OFFSETS):
        print("✅ Tokenised train corpus & offsets exist. Skipping.")
    else:
        print("🔤 Encoding training corpus...")
        start = time.time()

        tokenizer = Tokenizer.from_file(cfg.TOKENIZER_FILE)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")

        offsets      = []
        total_tokens = 0

        with open(cfg.TRAIN_TOKENS, "wb", buffering=1024 * 1024) as f_out, \
             open(cfg.TRAIN_JSONL,  "r",  encoding="utf-8") as f_in:

            for line in tqdm(f_in, desc="Encoding train documents", unit="doc"):
                start_idx = os.path.getsize(cfg.TRAIN_TOKENS) // 2
                text      = json.loads(line)["text"]
                ids       = tokenizer.encode(text).ids
                buffer    = np.array([bos] + ids + [eos], dtype=np.uint16)
                buffer.tofile(f_out)
                total_tokens += len(buffer)
                offsets.append({"start": start_idx, "length": len(buffer)})

        with open(cfg.DOC_OFFSETS, "w") as f:
            json.dump(offsets, f)

        print(f"✅ Train encoding done in {time.time() - start:.1f}s")
        print(f"📊 Total tokens: {total_tokens / 1_000_000:.2f}M ({total_tokens:,})")
        print(f"📄 Total documents: {len(offsets)}")

    # ── ✅ FIX: Test corpus ────────────────────────────────────────────
    # Previously test_raw.jsonl was saved by the downloader but NEVER encoded.
    # Perplexity evaluation MUST run on this held-out data, not on the
    # training stage bins — otherwise we are measuring training loss, not
    # generalisation.
    if os.path.exists(cfg.TEST_TOKENS):
        print("✅ Tokenised test corpus exists. Skipping.")
    else:
        print("🔤 Encoding test corpus...")
        start = time.time()
        n_tokens = _encode_file(
            cfg.TEST_JSONL,
            cfg.TEST_TOKENS,
            desc="Encoding test documents",
        )
        print(f"✅ Test encoding done in {time.time() - start:.1f}s")
        print(f"📊 Test tokens: {n_tokens / 1_000_000:.2f}M ({n_tokens:,})")