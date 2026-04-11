"""
3_encoder.py
Encodes the full training corpus into a single flat uint16 binary file.
Also saves document boundaries for fast stage-splitting later without re-tokenizing.
"""
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import Dataset.config as cfg

def encode_corpus():
    """Tokenizes train.jsonl -> train_tokens.bin + doc_offsets.json. Skips if exists."""
    import time
    start = time.time()
    if os.path.exists(cfg.TRAIN_TOKENS) and os.path.exists(cfg.DOC_OFFSETS):
        print("✅ Tokenized corpus & offsets exist. Skipping.")
        return

    print("🔤 Encoding training corpus...")
    tokenizer = Tokenizer.from_file(cfg.TOKENIZER_FILE)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    offsets = []

    total_tokens = 0
    
    with open(cfg.TRAIN_TOKENS, "wb", buffering=1024*1024) as f_out, \
         open(cfg.TRAIN_JSONL, "r", encoding="utf-8") as f_in:
        
        for line in tqdm(f_in, desc="Encoding documents", unit="doc"):
            start_idx = os.path.getsize(cfg.TRAIN_TOKENS) // 2
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text).ids
            
            # Write <bos> + ids + <eos>
            buffer = np.array([bos] + ids + [eos], dtype=np.uint16)
            buffer.tofile(f_out)

            total_tokens += len(buffer)
            
            # This is matedata of each document what is starting index of that document and what is lenght of that documnet.
            offsets.append({"start": start_idx, "length": len(buffer)})
            
    # Write that metadata of documents into one file properly
    with open(cfg.DOC_OFFSETS, "w") as f:
        json.dump(offsets, f)
        
    print(f"✅ Encoding complete in {time.time() - start:.1f}s")
    total_tokens_m = total_tokens / 1_000_000

    print(f"📊 Total tokens: {total_tokens_m:.2f}M ({total_tokens:,})")
    print(f"📄 Total documents: {len(offsets)}")