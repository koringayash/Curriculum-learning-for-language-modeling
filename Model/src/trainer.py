"""trainer.py: Main training loop with grad accumulation, resume, timing, checkpoints."""
import os
import sys
import time
import math
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Model.config as cfg
from .model import GPT
from .dataset import TokenDataset
from .curriculum_sampler import get_batch, describe_weights

def train(mode: str = "curriculum", resume: bool = False):
    print(f"🚀 Starting Training | Mode: {mode} | Resume: {resume}")
    config_global.set_seed()
    # cfg_global.DEVICE = config_global.DEVICE  # Make accessible to sampler
    cfg.USE_CURRICULUM = (mode == "curriculum")
    
    state_file = os.path.join(cfg.CHECKPOINT_DIR, f"train_state_{mode}.json")
    datasets = [TokenDataset(cfg.STAGE_BINS[i], cfg.SEQ_LEN) for i in range(1, 5)]
    
    total_seqs = sum(len(d) for d in datasets)
    steps_per_epoch = total_seqs // (cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * cfg.EPOCHS
    save_points = {int(total_steps * 0.2): "20%", int(total_steps * 0.4): "40%", 
                   int(total_steps * 0.6): "60%", int(total_steps * 0.8): "80%", total_steps: "100%"}
    
    model = GPT(cfg.VOCAB_SIZE, cfg.N_EMBD, cfg.N_HEAD, cfg.N_LAYER, cfg.SEQ_LEN, cfg.DROPOUT).to(config_global.DEVICE)

    # --- Model stats ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    model_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 ** 2)

    print(f"🧠 Total parameters: {total_params/1e6:.2f}M ({total_params:,})")
    print(f"🎯 Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"💾 Model size: {model_size_mb:.2f} MB")
        
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.dim() >= 2], "weight_decay": cfg.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if p.dim() < 2], "weight_decay": 0.0}
    ], lr=cfg.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.1*cfg.LEARNING_RATE)
    scaler = GradScaler(enabled=config_global.DEVICE=="cuda")
    rng = np.random.RandomState(config_global.RANDOM_SEED)
    
    start_step = 0
    if resume and os.path.exists(state_file):
        with open(state_file) as f: start_step = json.load(f)["step"]
        print(f"🔄 Resuming from step {start_step}")
        # Load latest checkpoint
        ckpt_files = sorted([f for f in os.listdir(cfg.CHECKPOINT_DIR) if f.startswith(f"ckpt_{mode}")], 
                            key=lambda x: int(x.split("step")[-1].split(".")[0]))
        if ckpt_files:
            latest = os.path.join(cfg.CHECKPOINT_DIR, ckpt_files[-1])
            ckpt = torch.load(latest, map_location=config_global.DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            # Restore RNG
            rng.set_state(np.load(os.path.join(cfg.CHECKPOINT_DIR, f"rng_{mode}.npy"), allow_pickle=True).item())
            
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    model.train()
    global_step = start_step
    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        
        for step in pbar:
            if global_step < start_step:
                global_step += 1
                continue
                
            x, y = get_batch(datasets, cfg.BATCH_SIZE, global_step/total_steps, rng)
            
            with autocast(enabled=config_global.DEVICE=="cuda"):
                _, loss = model(x, y)
                loss = loss / cfg.ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
            scheduler.step()
            epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS
            
            if global_step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item()*cfg.ACCUMULATION_STEPS:.4f}", 
                                  "lr": f"{scheduler.get_last_lr()[0]:.2e}", "mix": describe_weights(global_step/total_steps)})
                
            # Checkpoint
            if global_step in save_points:
                tag = f"ckpt_{mode}_{save_points[global_step]}_step{global_step}.pt"
                torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, 
                           os.path.join(cfg.CHECKPOINT_DIR, tag))
                np.save(os.path.join(cfg.CHECKPOINT_DIR, f"rng_{mode}.npy"), rng.get_state())
                with open(state_file, "w") as f: json.dump({"step": global_step}, f)
                print(f"\n💾 Checkpoint saved: {save_points[global_step]} ({global_step} steps)")
                
    print(f"\n✅ Training complete. Total time tracked via tqdm & checkpoints.")

if __name__ == "__main__":
    train()