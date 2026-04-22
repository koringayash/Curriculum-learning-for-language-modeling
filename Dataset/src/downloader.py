"""
1_downloader.py
Streams FineWeb, applies a single-pass prose quality filter, and saves 
train/test JSONL files. Skips if files already exist.
"""
import os
import sys
import json
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import Dataset.config as cfg


sys.stdout = open('logs.txt', 'a')

def is_quality_prose(text: str) -> bool:
    """Single-pass prose quality filter. Returns True if text passes all checks."""
    if len(text) < cfg.MIN_DOC_CHARS:
        return False
    
    non_space = alpha = digit = 0
    for c in text:
        if not c.isspace():
            non_space += 1
            if c.isalpha(): alpha += 1
            elif c.isdigit(): digit += 1
            
    if non_space == 0: return False
    if (alpha / non_space) < 0.70: return False
    if (digit / len(text)) >= 0.15: return False
    
    words = text.split()
    if len(words) < 50: return False
    
    stops = {"the","is","in","and","to","of","a","that","it","was","for","on","are","with","as","at","be","this"}
    if sum(1 for w in words if w.lower() in stops) / len(words) < 0.05: return False
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 3 and sum(1 for l in lines if len(l) < 20) / len(lines) >= 0.60:
        return False
        
    return True

def download_and_split():
    """Stream FineWeb, filter prose, reserve test docs, save remaining as train."""
    import time
    start = time.time()
    if os.path.exists(cfg.TRAIN_JSONL) and os.path.exists(cfg.TEST_JSONL):
        print("✅ Dataset files already exist. Skipping download.")
        return

    from datasets import load_dataset
    print("🌐 Streaming FineWeb dataset...")
    # This will download dataset from huggingface.
    dataset = load_dataset(cfg.FINEWEB_DATASET, name=cfg.FINEWEB_CONFIG, split="train", streaming=True)
    
    os.makedirs(os.path.dirname(cfg.TRAIN_JSONL), exist_ok=True)
    
    with open(cfg.TEST_JSONL, "w", encoding="utf-8") as f_test, \
         open(cfg.TRAIN_JSONL, "w", encoding="utf-8") as f_train:
        
        test_n = train_n = 0
        total_needed = cfg.TEST_DOCS + cfg.TRAIN_DOCS + 2000  # buffer for filtering
        
        # Here dataset is iterator
        pbar = tqdm(dataset, desc="Downloading & Filtering", total=total_needed, unit="doc")
        for sample in pbar:
            if test_n >= cfg.TEST_DOCS and train_n >= cfg.TRAIN_DOCS:
                print(f"We have Already downloaded enough data for training and testing.")
                break
                
            text = sample.get("text", "").strip()
            if not is_quality_prose(text):
                continue
                
            record = json.dumps({"id": sample.get("id", ""), "text": text}, ensure_ascii=False)
            if test_n < cfg.TEST_DOCS:
                f_test.write(record + "\n")
                test_n += 1
            else:
                f_train.write(record + "\n")
                train_n += 1
                
            pbar.set_postfix({"train": train_n, "test": test_n})
            
    print(f"✅ Download complete in {time.time() - start:.1f}s | Train: {train_n} | Test: {test_n}")

    train_size_mb = os.path.getsize(cfg.TRAIN_JSONL) / (1024 * 1024)
    test_size_mb = os.path.getsize(cfg.TEST_JSONL) / (1024 * 1024)

    print(f"📦 Train file size: {train_size_mb:.2f} MB")
    print(f"📦 Test file size: {test_size_mb:.2f} MB")