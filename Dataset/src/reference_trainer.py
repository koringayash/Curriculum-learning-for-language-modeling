"""
4_reference_trainer.py
Trains a tiny reference GPT on the tokenized corpus.
Imports GPT from Model module to avoid duplication. Used only for scoring.
"""
import os
import sys
import math
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Dataset.config as cfg
from Model.src.model import GPT
from Model.src.dataset import TokenDataset

sys.stdout = open('logs.txt', 'a')

def train_reference_model():
    """Trains tiny reference model for 1 epoch. Skips if checkpoint exists."""
    import time
    start = time.time()
    ckpt_path = os.path.join(config_global.PROJECT_ROOT, "checkpoints", "reference_model.pt")
    if os.path.exists(ckpt_path):
        print("✅ Reference model exists. Skipping.")
        return

    print("🧠 Training reference model for scoring...")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    config_global.set_seed(cfg.RANDOM_SEED)
    
    # Making (x,y) pairs for traning 
    dataset = TokenDataset(cfg.TRAIN_TOKENS, cfg.REF_SEQ_LEN)
    # This will create batch from binary files of ids of fixed length 
    loader = DataLoader(dataset, batch_size=cfg.REF_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    
    model = GPT(
        vocab_size=cfg.VOCAB_SIZE, n_embd=cfg.REF_N_EMBD, n_head=cfg.REF_N_HEAD,
        n_layer=cfg.REF_N_LAYER, seq_len=cfg.REF_SEQ_LEN, dropout=0.0
    ).to(config_global.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.REF_LR, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=config_global.DEVICE=="cuda")
    model.train()
    
    for epoch in range(1, cfg.REF_EPOCHS + 1):
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"Ref Model Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(config_global.DEVICE, non_blocking=True), y.to(config_global.DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=config_global.DEVICE=="cuda"):
                _, loss = model(x, y)
                
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            total_loss += loss_val
            n += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "ppl": f"{math.exp(min(loss_val, 10)):.2f}"})
            
    avg_loss = total_loss / n
    torch.save({
        "model_state": model.state_dict(),
        "arch": {"vocab_size": cfg.VOCAB_SIZE, "n_embd": cfg.REF_N_EMBD, 
                 "n_head": cfg.REF_N_HEAD, "n_layer": cfg.REF_N_LAYER, "seq_len": cfg.REF_SEQ_LEN}
    }, ckpt_path)
    print(f"✅ Reference model saved in {time.time() - start:.1f}s | Avg Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 10)):.2f}")