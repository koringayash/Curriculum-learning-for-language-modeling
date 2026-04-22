"""trainer.py: Main training loop with grad accumulation, resume, timing, checkpoints.

Curriculum mode  → Epoch-staged curriculum. Each epoch trains on a growing
                   subset of difficulty stages (defined in config.CURRICULUM_EPOCH_STAGES).
                   Every document in every included stage is visited exactly once per epoch
                   (DataLoader with shuffle=True, drop_last=True) — no probabilistic sampling,
                   no documents skipped.

                   Default (4 stages, 3 epochs):
                     Epoch 1  →  Stage 1 only          (~25k docs)
                     Epoch 2  →  Stages 1 + 2           (~50k docs)
                     Epoch 3  →  Stages 1 + 2 + 3 + 4   (~100k docs)

                   Final artefact: curriculum_final.pt

Random mode      → All 4 stages, all epochs, shuffled uniformly.
                   Every document seen once per epoch — same coverage as
                   curriculum epoch 3, but held constant throughout.
                   Final artefact: random_final.pt
"""
import os
import sys
import json
import numpy as np
import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Model.config as cfg
from .model import GPT
from .dataset import TokenDataset

sys.stdout = open('logs.txt', 'a')
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_model_stats(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes    = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb      = (param_bytes + buf_bytes) / (1024 ** 2)
    print(f"🧠 Total parameters    : {total_params/1e6:.2f}M ({total_params:,})")
    print(f"🎯 Trainable parameters: {trainable/1e6:.2f}M")
    print(f"💾 Model size          : {size_mb:.2f} MB")


def _make_loader(stage_ids: list, shuffle: bool) -> DataLoader:
    """Build a DataLoader that covers all documents in the given stage bins."""
    datasets = [TokenDataset(cfg.STAGE_BINS[s], cfg.SEQ_LEN) for s in stage_ids]
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        combined,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=(config_global.DEVICE == "cuda"),
        num_workers=2,
        drop_last=True,
    )


def _compute_total_steps(epoch_stage_map: dict) -> int:
    """
    Add up gradient steps across all epochs so the cosine LR scheduler
    spans the entire training run correctly.
    Note: each epoch may have a different number of steps because the
    dataset grows as more stages are added.
    """
    total = 0
    for stage_ids in epoch_stage_map.values():
        loader = _make_loader(stage_ids, shuffle=False)
        total += len(loader) // cfg.ACCUMULATION_STEPS
    return total


def _build_model_and_optimizer(total_steps: int):
    model = GPT(
        cfg.VOCAB_SIZE, cfg.N_EMBD, cfg.N_HEAD,
        cfg.N_LAYER, cfg.SEQ_LEN, cfg.DROPOUT,
    ).to(config_global.DEVICE)
    _print_model_stats(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if p.dim() >= 2],
             "weight_decay": cfg.WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters() if p.dim() < 2],
             "weight_decay": 0.0},
        ],
        lr=cfg.LEARNING_RATE, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.1 * cfg.LEARNING_RATE,
    )

    # When you train models using float16 (FP16) instead of full precision (FP32): gradients can become too small (underflow) → training breaks
    # It prevents those tiny gradients from vanishing by scaling them up temporarily.
    
    scaler = GradScaler(enabled=(config_global.DEVICE == "cuda"))
    return model, optimizer, scheduler, scaler


def _save_checkpoint(path, model, optimizer, scheduler, scaler, global_step):
    torch.save(
        {
            "model_state"    : model.state_dict(),
            "optim_state"    : optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state"   : scaler.state_dict(),
            "global_step"    : global_step,
        },
        path,
    )
    print(f"\n💾 Checkpoint saved → {os.path.basename(path)}")


def _load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=config_global.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step = ckpt.get("global_step", 0)
    print(f"🔄 Resumed from {os.path.basename(path)}  (step {step})")
    return step


def _delete_if_exists(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def _rename_checkpoint(src: str, dst: str) -> None:
    if src and os.path.exists(src):
        os.rename(src, dst)


# ---------------------------------------------------------------------------
# One shared epoch loop  (used by both training modes)
# ---------------------------------------------------------------------------

def _run_epoch(model, optimizer, scheduler, scaler, loader, epoch, total_epochs, global_step):
    """Iterate one full DataLoader epoch. Returns (global_step, avg_loss)."""
    model.train()
    epoch_loss  = 0.0
    micro_count = 0
    loss_list = []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", dynamic_ncols=True)

    for micro_step, (x, y) in enumerate(pbar):
        x = x.to(config_global.DEVICE, non_blocking=True)
        y = y.to(config_global.DEVICE, non_blocking=True)

        with autocast(enabled=(config_global.DEVICE == "cuda")):
            _, loss = model(x, y)
            loss = loss / cfg.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (micro_step + 1) % cfg.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS
        micro_count += 1

        if global_step % 10 == 0:
            loss_list.append(loss.item() * cfg.ACCUMULATION_STEPS)
            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}",
                "lr"  : f"{scheduler.get_last_lr()[0]:.2e}",
            })

    avg_loss = epoch_loss / max(micro_count, 1)
    return global_step, avg_loss, loss_list


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------

def _train_curriculum(resume: bool) -> None:
    print("🚀 Curriculum Training  |  Epoch-staged  |  Full coverage per epoch")
    config_global.set_seed()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    epoch_stages = cfg.CURRICULUM_EPOCH_STAGES   # {1:[1], 2:[1,2], 3:[1,2,3,4]}
    total_steps  = _compute_total_steps(epoch_stages)
    state_file   = os.path.join(cfg.CHECKPOINT_DIR, "train_state_curriculum.json")

    print(f"📐 Total grad steps across all epochs: {total_steps:,}")
    for ep, stages in epoch_stages.items():
        loader = _make_loader(stages, shuffle=False)
        steps  = len(loader) // cfg.ACCUMULATION_STEPS
        n_docs = sum(len(TokenDataset(cfg.STAGE_BINS[s], cfg.SEQ_LEN)) for s in stages)
        print(f"   Epoch {ep}: stages {stages}  |  {n_docs:,} docs  |  {steps:,} grad steps")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    prev_ckpt   = None

    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state.get("epoch", 0)
        ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"curriculum_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            global_step = _load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
            prev_ckpt   = ckpt_path
            print(f"   Resuming from epoch {start_epoch + 1}")
        else:
            print("⚠️  Checkpoint not found. Starting fresh.")
            start_epoch = 0

    # ── main loop ─────────────────────────────────────────────────────────
    for epoch_num, stage_ids in epoch_stages.items():
        if epoch_num <= start_epoch:
            continue                              # skip already-completed epochs

        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch_num}/{cfg.EPOCHS}  |  Active stages: {stage_ids}")
        print(f"{'='*60}")

        loader = _make_loader(stage_ids, shuffle=True)
        global_step, avg_loss, loss_list = _run_epoch(
            model, optimizer, scheduler, scaler,
            loader, epoch_num, cfg.EPOCHS, global_step,
        )

        loss_file = os.path.join(cfg.CHECKPOINT_DIR, "loss_all.json")

        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append({
            "mode":"curriculum",
            "epoch": epoch_num,
            "loss": loss_list
        })

        with open(loss_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n   ✅ Epoch {epoch_num} done | "
              f"avg loss: {avg_loss:.4f} | "
              f"time: {time.time() - epoch_start:.1f}s")

        cur_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"curriculum_epoch{epoch_num}.pt")
        _save_checkpoint(cur_ckpt, model, optimizer, scheduler, scaler, global_step)
        _delete_if_exists(prev_ckpt)
        prev_ckpt = cur_ckpt

        with open(state_file, "w") as f:
            json.dump({"epoch": epoch_num, "global_step": global_step}, f)

    final_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curriculum_final.pt")
    _rename_checkpoint(prev_ckpt, final_ckpt)

    print("\n✅ Curriculum training complete!")
    print("📁 Final checkpoint → curriculum_final.pt")


# ---------------------------------------------------------------------------
# Random training
# ---------------------------------------------------------------------------

def _train_random(resume: bool) -> None:
    print("🚀 Random Training  |  All stages every epoch  |  Fresh shuffle each time")
    config_global.set_seed()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    all_stages   = [1, 2, 3, 4]
    one_ep_steps = len(_make_loader(all_stages, shuffle=True)) // cfg.ACCUMULATION_STEPS
    total_steps  = one_ep_steps * cfg.EPOCHS
    state_file   = os.path.join(cfg.CHECKPOINT_DIR, "train_state_random.json")

    print(f"📐 Steps/epoch: {one_ep_steps:,}  |  Total steps: {total_steps:,}")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    prev_ckpt   = None

    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state.get("epoch", 0)
        ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            global_step = _load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
            prev_ckpt   = ckpt_path
            print(f"🔄 Resuming from epoch {start_epoch + 1}/{cfg.EPOCHS}")
        else:
            print("⚠️  Checkpoint not found. Starting fresh.")
            start_epoch = 0

    # ── main loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{cfg.EPOCHS}  |  All stages, uniform shuffle")
        print(f"{'='*60}")

        loader = _make_loader(all_stages, shuffle=True)   # fresh shuffle each epoch
        global_step, avg_loss, loss_list = _run_epoch(
            model, optimizer, scheduler, scaler,
            loader, epoch, cfg.EPOCHS, global_step,
        )

        
        loss_file = os.path.join(cfg.CHECKPOINT_DIR, "loss_all.json")

        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append({
            "mode":"random",
            "epoch": epoch,
            "loss": loss_list
        })

        with open(loss_file, "w") as f:
            json.dump(data, f, indent=2)


        print(f"\n   ✅ Epoch {epoch} done | "
              f"avg loss: {avg_loss:.4f} | "
              f"time: {time.time() - epoch_start:.1f}s")

        cur_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{epoch}.pt")
        _save_checkpoint(cur_ckpt, model, optimizer, scheduler, scaler, global_step)
        _delete_if_exists(prev_ckpt)
        prev_ckpt = cur_ckpt

        with open(state_file, "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step}, f)

    final_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "random_final.pt")
    _rename_checkpoint(prev_ckpt, final_ckpt)

    print("\n✅ Random training complete!")
    print("📁 Final checkpoint → random_final.pt")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(mode: str = "curriculum", resume: bool = False) -> None:
    print(f"🚀 Starting Training | Mode: {mode} | Resume: {resume}")
    if mode == "curriculum":
        _train_curriculum(resume)
    elif mode == "random":
        _train_random(resume)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'curriculum' or 'random'.")


if __name__ == "__main__":
    train()