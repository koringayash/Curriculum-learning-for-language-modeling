"""trainer.py: Main training loop with grad accumulation, resume, timing, checkpoints.

Curriculum mode  → Original gradual-drift approach. One continuous training run
                   where sampling weights shift smoothly from Stage-1-heavy to
                   Stage-4-heavy as progress goes 0% → 100%.
                   Checkpoints saved at 25%, 50%, 75%, 100% of total steps.
                   All 4 are kept as training-progress snapshots.
                   Final artefacts: curriculum_25pct.pt, curriculum_50pct.pt,
                                    curriculum_75pct.pt, curriculum_final.pt

Random mode      → EPOCHS epochs with equal 25% weights across all datasets.
                   Saves after every epoch, deletes previous (crash-recovery).
                   Resume picks up from the last completed epoch automatically.
                   Final artefact: random_final.pt

Total checkpoints after a full experiment: 5 models.
"""
import os
import sys
import json
import numpy as np
import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Model.config as cfg
from .model import GPT
from .dataset import TokenDataset
from .curriculum_sampler import get_batch, describe_weights

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_model_stats(model: torch.nn.Module) -> None:
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes      = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes        = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb          = (param_bytes + buf_bytes) / (1024 ** 2)
    print(f"🧠 Total parameters    : {total_params/1e6:.2f}M ({total_params:,})")
    print(f"🎯 Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"💾 Model size          : {size_mb:.2f} MB")


def _build_model_and_optimizer(total_steps: int):
    """Instantiate GPT, AdamW, cosine-annealing scheduler, and AMP GradScaler."""
    model = GPT(
        cfg.VOCAB_SIZE, cfg.N_EMBD, cfg.N_HEAD,
        cfg.N_LAYER, cfg.SEQ_LEN, cfg.DROPOUT
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
        optimizer, T_max=total_steps, eta_min=0.1 * cfg.LEARNING_RATE
    )
    scaler = GradScaler(enabled=(config_global.DEVICE == "cuda"))
    return model, optimizer, scheduler, scaler


def _save_checkpoint(path, model, optimizer, scheduler, scaler, rng, global_step):
    """Save full training state: model + optimizer + scheduler + scaler + RNG + step."""
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

    # ✅ FIX: store RNG safely
    rng_state = rng.get_state()
    np.save(
        path.replace(".pt", "_rng.npy"),
        np.array(rng_state, dtype=object)   # <-- FIX HERE
    )

    print(f"\n💾 Checkpoint saved → {os.path.basename(path)}")


def _load_checkpoint(path, model, optimizer, scheduler, scaler, rng):
    """Restore full training state. Returns the saved global_step."""
    ckpt = torch.load(path, map_location=config_global.DEVICE)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])

    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    # ✅ FIX: load RNG safely
    rng_path = path.replace(".pt", "_rng.npy")
    if os.path.exists(rng_path):
        rng_state = np.load(rng_path, allow_pickle=True)
        rng.set_state(tuple(rng_state))   # <-- FIX HERE

    step = ckpt.get("global_step", 0)

    print(f"🔄 Resumed from checkpoint → {os.path.basename(path)}  (step {step})")
    return step


def _delete_if_exists(path: str) -> None:
    """Remove a checkpoint file and its companion RNG .npy file if they exist."""
    if not path:
        return
    if os.path.exists(path):
        os.remove(path)
    rng_path = path.replace(".pt", "_rng.npy")
    if os.path.exists(rng_path):
        os.remove(rng_path)


def _rename_checkpoint(src: str, dst: str) -> None:
    """Rename a checkpoint file and its companion RNG file."""
    if src and os.path.exists(src):
        os.rename(src, dst)
    src_rng, dst_rng = src.replace(".pt", "_rng.npy"), dst.replace(".pt", "_rng.npy")
    if os.path.exists(src_rng):
        os.rename(src_rng, dst_rng)


# ---------------------------------------------------------------------------
# Curriculum training  (gradual-drift, single continuous run)
# ---------------------------------------------------------------------------

def _train_curriculum(resume: bool) -> None:
    """
    Original gradual-drift curriculum learning.

    One continuous training loop over all epochs. The progress ratio
    (global_step / total_steps) is passed to get_batch() on every step,
    which calls get_curriculum_weights() to smoothly shift the sampling
    distribution from Stage-1-heavy (progress=0) to Stage-4-heavy (progress=1).

    Checkpoints are saved at 25 / 50 / 75 / 100 % of total gradient steps.
    All four are kept on disk as training-progress snapshots:
        curriculum_25pct.pt
        curriculum_50pct.pt
        curriculum_75pct.pt
        curriculum_final.pt

    Resume:
        Reads train_state_curriculum.json → finds the latest checkpoint →
        restores model + optimizer + scheduler + scaler + RNG state →
        skips already-completed loop iterations WITHOUT replaying them
        (the RNG is already in the correct state from the saved checkpoint).
    """
    print("🚀 Curriculum Training  |  Gradual-drift weights  |  Saves at 25/50/75/100%")
    config_global.set_seed()
    cfg.USE_CURRICULUM = True
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    state_file = os.path.join(cfg.CHECKPOINT_DIR, "train_state_curriculum.json")

    datasets = [TokenDataset(cfg.STAGE_BINS[i], cfg.SEQ_LEN) for i in range(1, 5)]
    total_seqs    = sum(len(d) for d in datasets)
    steps_per_epoch = total_seqs // (cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS)
    total_steps   = steps_per_epoch * cfg.EPOCHS

    # Save-point gradient steps (at exactly 25 / 50 / 75 / 100 %)
    save_points = {
        int(total_steps * 0.25): "25pct",
        int(total_steps * 0.50): "50pct",
        int(total_steps * 0.75): "75pct",
        total_steps             : "final",
    }
    # Map label → filename for reference after training
    ckpt_names = {
        label: os.path.join(cfg.CHECKPOINT_DIR, f"curriculum_{label}.pt")
        for label in save_points.values()
    }

    print(f"📐 Steps/epoch: {steps_per_epoch:,}  |  Total steps: {total_steps:,}")
    print(f"   Save points (grad steps): { {v: k for k, v in save_points.items()} }")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)
    rng = np.random.RandomState(config_global.RANDOM_SEED)

    # ── resume ────────────────────────────────────────────────────────────
    start_step = 0
    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        saved_step  = state.get("global_step", 0)
        saved_label = state.get("label", "")

        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"curriculum_{saved_label}.pt")
        if os.path.exists(ckpt_path):
            start_step = _load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler, rng
            )
        else:
            print(f"⚠️  Checkpoint not found ({os.path.basename(ckpt_path)}). Starting fresh.")
            start_step = 0

    # Compute which epoch and micro-step to resume from so we can skip
    # already-completed iterations without touching the RNG
    # (RNG was restored from the checkpoint → already in the right state)
    resume_epoch = start_step // steps_per_epoch
    resume_micro = (start_step % steps_per_epoch) * cfg.ACCUMULATION_STEPS

    # ── main loop ─────────────────────────────────────────────────────────
    global_step = start_step
    model.train()

    for epoch in range(cfg.EPOCHS):

        epoch_start = time.time()

        # Skip fully completed epochs without touching RNG
        if epoch < resume_epoch:
            continue

        start_micro = resume_micro if epoch == resume_epoch else 0
        total_micro = steps_per_epoch * cfg.ACCUMULATION_STEPS
        pbar = tqdm(
            range(start_micro, total_micro),
            desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}",
            dynamic_ncols=True,
        )
        epoch_loss = 0.0

        if start_micro > 0:
            optimizer.zero_grad(set_to_none=True)  # ensure clean state on partial-epoch resume

        for micro_step in pbar:
            progress = global_step / total_steps   # 0.0 → 1.0 across the full run

            x, y = get_batch(datasets, cfg.BATCH_SIZE, progress, rng)

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

                # ── checkpoint ────────────────────────────────────────────
                if global_step in save_points:
                    label    = save_points[global_step]
                    ckpt_out = ckpt_names[label]
                    _save_checkpoint(
                        ckpt_out, model, optimizer, scheduler, scaler, rng, global_step
                    )
                    with open(state_file, "w") as f:
                        json.dump({"global_step": global_step, "label": label}, f)
                    pct = label.replace("pct", "%").replace("final", "100%")
                    print(f"   📍 {pct} of training complete  ({global_step:,} / {total_steps:,} steps)")

            epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS

            if global_step % 10 == 0:
                pbar.set_postfix({
                    "loss"    : f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}",
                    "lr"      : f"{scheduler.get_last_lr()[0]:.2e}",
                    "progress": f"{global_step/total_steps*100:.1f}%",
                    "mix"     : describe_weights(global_step / total_steps),
                })

        avg_loss = epoch_loss / (total_micro - start_micro)
        print(f"\n   ✅ Epoch {epoch + 1}/{cfg.EPOCHS} done | avg loss: {avg_loss:.4f}")
        resume_micro = 0   # only the first resumed epoch has a non-zero start_micro
        print(f"📦 Epoch finished in {time.time() - epoch_start:.2f}s")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n✅ Curriculum training complete!")
    print("📁 Saved checkpoints:")
    for label, path in ckpt_names.items():
        tag = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"   {tag}  {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Random training  (equal weights, epoch-level crash-recovery checkpoints)
# ---------------------------------------------------------------------------

def _train_random(resume: bool) -> None:
    """
    Train with equal 25% sampling across all 4 stage datasets.

    A checkpoint is written after every epoch; the previous one is deleted,
    so at most ONE checkpoint lives on disk at any time.
    If training crashes mid-epoch you lose that epoch and resume from the
    last fully completed one.

    Final checkpoint (1 file): random_final.pt
    """
    print("🚀 Random Training  |  Equal 25% weights across all stages")
    config_global.set_seed()
    cfg.USE_CURRICULUM = False
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    state_file = os.path.join(cfg.CHECKPOINT_DIR, "train_state_random.json")

    datasets    = [TokenDataset(cfg.STAGE_BINS[i], cfg.SEQ_LEN) for i in range(1, 5)]
    total_seqs  = sum(len(d) for d in datasets)
    steps_ep    = total_seqs // (cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS)
    total_steps = steps_ep * cfg.EPOCHS
    weights     = np.array([0.25, 0.25, 0.25, 0.25])

    print(f"📐 Steps/epoch: {steps_ep:,}  |  Total steps: {total_steps:,}")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)
    rng = np.random.RandomState(config_global.RANDOM_SEED)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    prev_ckpt   = None

    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state.get("epoch", 0)
        ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            _load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, rng)
            prev_ckpt = ckpt_path
            print(f"🔄 Resuming from epoch {start_epoch + 1}/{cfg.EPOCHS}")
        else:
            print(f"⚠️  Checkpoint not found ({os.path.basename(ckpt_path)}). Starting fresh.")
            start_epoch = 0

    # ── main loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"🎲 Random training | {steps_ep:,} gradient steps/epoch | {cfg.EPOCHS} epochs")
    print(f"   Starting from epoch: {start_epoch + 1}")
    print(f"{'='*62}")

    model.train()
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_loss  = 0.0
        total_micro = steps_ep * cfg.ACCUMULATION_STEPS
        pbar = tqdm(
            range(total_micro),
            desc=f"Random | Epoch {epoch + 1}/{cfg.EPOCHS}",
            dynamic_ncols=True,
        )
        global_step = epoch * steps_ep
        optimizer.zero_grad(set_to_none=True)

        for micro_step in pbar:
            weights_norm = weights / weights.sum()
            xs, ys = [], []
            for _ in range(cfg.BATCH_SIZE):
                si = rng.choice(4, p=weights_norm)
                ds = datasets[si]
                ii = rng.randint(0, len(ds))
                x, y = ds[ii]
                xs.append(x); ys.append(y)
            x = torch.stack(xs).to(config_global.DEVICE)
            y = torch.stack(ys).to(config_global.DEVICE)

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

            if global_step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}",
                    "lr"  : f"{scheduler.get_last_lr()[0]:.2e}",
                })

        avg_loss = epoch_loss / total_micro
        print(f"\n   ✅ Epoch {epoch + 1}/{cfg.EPOCHS} done | avg loss: {avg_loss:.4f}")

        # Save this epoch's checkpoint
        cur_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{epoch + 1}.pt")
        _save_checkpoint(cur_ckpt, model, optimizer, scheduler, scaler, rng, global_step)

        # Delete the previous epoch's checkpoint to save disk space
        _delete_if_exists(prev_ckpt)
        prev_ckpt = cur_ckpt

        # Persist training state for crash-recovery
        with open(state_file, "w") as f:
            json.dump({"epoch": epoch + 1}, f)

    # Rename last epoch checkpoint → random_final.pt
    final_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "random_final.pt")
    _rename_checkpoint(prev_ckpt, final_ckpt)

    print("\n✅ Random training complete!")
    print(f"📁 Final checkpoint → random_final.pt")


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