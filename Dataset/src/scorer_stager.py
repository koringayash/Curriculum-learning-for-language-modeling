"""
5_scorer_stager.py
Scores documents using reference model + readability metrics.
Splits into 4 difficulty stages. Copies token slices from pre-encoded corpus.
"""
import os
import sys
import json
import gzip
import math
import numpy as np
import torch
import textstat
from tqdm import tqdm
from tokenizers import Tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Dataset.config as cfg
from Model.src.model import GPT

@torch.no_grad()
def compute_ppl(text: str, model: GPT, tokenizer: Tokenizer) -> float:
    """Compute document perplexity using reference model (batched chunks)."""
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    ids = [bos] + tokenizer.encode(text).ids + [eos]
    if len(ids) < 4: return 1.0
    
    total_nll, total_n = 0.0, 0
    for i in range(0, len(ids) - 1, cfg.REF_SEQ_LEN):
        chunk = ids[i:i + cfg.REF_SEQ_LEN + 1]
        if len(chunk) < 2: continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=config_global.DEVICE)
        y = torch.tensor([chunk[1:]],  dtype=torch.long, device=config_global.DEVICE)
        _, loss = model(x, y)
        total_nll += loss.item() * (len(chunk) - 1)
        total_n   += (len(chunk) - 1)
    return min(math.exp(total_nll / max(total_n, 1)), 10000.0)

def score_documents(docs: list, model: GPT, tokenizer: Tokenizer) -> np.ndarray:
    """Compute weighted difficulty scores for all documents."""
    n = len(docs)
    scores = np.zeros(n)
    pbar = tqdm(docs, desc="Scoring documents", unit="doc")
    
    for i, doc in enumerate(pbar):
        text = doc["text"]
        ppl  = compute_ppl(text, model, tokenizer)
        fk   = max(0, min(textstat.flesch_kincaid_grade(text), 20))
        ttr  = len(set(text.lower().split())) / max(len(text.split()), 1)
        comp = 1.0 - (len(gzip.compress(text.encode())) / max(len(text.encode()), 1))
        
        # Normalize later after collecting all
        doc["_raw"] = {"ppl": ppl, "fk": fk, "ttr": ttr, "comp": comp}
        
    # Global normalization
    metrics = ["ppl", "fk", "ttr", "comp"]
    weights = [cfg.SCORE_WEIGHT_PPL, cfg.SCORE_WEIGHT_FLESCH, cfg.SCORE_WEIGHT_TTR, cfg.SCORE_WEIGHT_COMP]
    
    raw = {m: np.array([d["_raw"][m] for d in docs]) for m in metrics}
    for m in metrics:
        lo, hi = raw[m].min(), raw[m].max()
        if hi > lo: raw[m] = (raw[m] - lo) / (hi - lo)
            
    for i in range(n):
        scores[i] = sum(weights[j] * raw[metrics[j]][i] for j in range(4))
    return scores

def stage_and_save(scores: np.ndarray, docs: list):
    """Sort by difficulty, split into quartiles, save JSONL + binary stages."""
    import time
    start = time.time()
    print("📦 Staging documents...")
    
    sorted_idx = np.argsort(scores)
    q = len(docs) // 4
    stage_indices = {
        1: sorted_idx[:q],
        2: sorted_idx[q:2*q],
        3: sorted_idx[2*q:3*q],
        4: sorted_idx[3*q:]
    }
    
    boundaries = {
        "q1": float(np.median(scores[stage_indices[1]])),
        "q2": float(np.median(scores[stage_indices[2]])),
        "q3": float(np.median(scores[stage_indices[3]]))
    }
    with open(cfg.STAGE_BOUNDARIES, "w") as f: json.dump(boundaries, f)
    
    # Load pre-encoded offsets
    with open(cfg.DOC_OFFSETS) as f: offsets = json.load(f)
    with open(cfg.TRAIN_TOKENS, "rb") as f_bin:
        all_tokens = np.memmap(f_bin, dtype=np.uint16, mode="r")
        
    for stage_id, idxs in stage_indices.items():
        # Save stage JSONL
        stage_jsonl = os.path.join(cfg.DATA_DIR, f"stage_{stage_id}.jsonl")
        with open(stage_jsonl, "w") as f:
            for i in idxs: json.dump(docs[i], f); f.write("\n")
            
        # Save stage Binary (slice from pre-encoded corpus)
        tokens = []
        for i in idxs:
            off = offsets[i]
            tokens.append(all_tokens[off["start"]:off["start"]+off["length"]])
        stage_bin = np.concatenate(tokens) if tokens else np.array([], dtype=np.uint16)
        stage_bin.tofile(cfg.STAGE_BINS[stage_id])
        
        print(f"  Stage {stage_id}: {len(idxs)} docs | {len(stage_bin):,} tokens")
        
    print(f"✅ Staging complete in {time.time() - start:.1f}s")

def run():
    """Orchestrates scoring and staging. Skips if stage bins exist."""
    import time
    start = time.time()
    if all(os.path.exists(p) for p in cfg.STAGE_BINS.values()):
        print("✅ Stage files exist. Skipping scoring/staging.")
        return
        
    print("🔍 Scoring & Staging pipeline...")
    with open(cfg.TRAIN_JSONL) as f: docs = [json.loads(l) for l in f]
    
    print("Loading reference model & tokenizer...")
    ckpt = torch.load(os.path.join(config_global.PROJECT_ROOT, "checkpoints", "reference_model.pt"), map_location=config_global.DEVICE)
    model = GPT(**ckpt["arch"], dropout=0.0).to(config_global.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(cfg.TOKENIZER_FILE)
    tokenizer.no_padding(); tokenizer.no_truncation()
    
    scores = score_documents(docs, model, tokenizer)
    stage_and_save(scores, docs)
    print(f"✅ Pipeline complete in {time.time() - start:.1f}s")