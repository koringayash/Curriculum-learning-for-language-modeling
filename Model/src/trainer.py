"""trainer.py: Main training loop with grad accumulation, resume, timing, checkpoints.

Curriculum mode  → 4 sequential stages, each trained for EPOCHS epochs using that
                   stage's blend weights. Saves after every epoch (deletes previous),
                   so disk always holds only the latest epoch per stage.
                   Final artefacts: curriculum_stage{1-4}_final.pt  (4 files)

Random mode      → EPOCHS epochs with equal 25% weights across all datasets.
                   Saves after every epoch (deletes previous) for crash-recovery.
                   Resume picks up from the last completed epoch automatically.
                   Final artefact: random_final.pt  (1 file)

Total checkpoints after a full experiment: 5 models.
"""
import os
import sys
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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_model_stats(model: torch.nn.Module) -> None:
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes      = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes        = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb          = (param_bytes + buf_bytes) / (1024 ** 2)
    print(f"🧠 Total parameters   : {total_params/1e6:.2f}M ({total_params:,})")
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


def _sample_batch(
    datasets: list,
    batch_size: int,
    weights: np.ndarray,
    rng: np.random.RandomState,
) -> tuple:
    """Sample a batch from the dataset list using the given probability weights."""
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    xs, ys = [], []
    for _ in range(batch_size):
        stage_idx  = rng.choice(len(datasets), p=weights)
        ds         = datasets[stage_idx]
        sample_idx = rng.randint(0, len(ds))
        x, y       = ds[sample_idx]
        xs.append(x)
        ys.append(y)
    return (
        torch.stack(xs).to(config_global.DEVICE),
        torch.stack(ys).to(config_global.DEVICE),
    )


def _save_checkpoint(
    path: str,
    model, optimizer, scheduler, scaler,
    rng: np.random.RandomState,
) -> None:
    """Save full training state: model + optimizer + scheduler + scaler + RNG."""
    torch.save(
        {
            "model_state"    : model.state_dict(),
            "optim_state"    : optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state"   : scaler.state_dict(),
        },
        path,
    )
    np.save(path.replace(".pt", "_rng.npy"), rng.get_state())
    print(f"\n💾 Checkpoint saved → {os.path.basename(path)}")


def _load_checkpoint(
    path: str,
    model, optimizer, scheduler, scaler,
    rng: np.random.RandomState,
) -> None:
    """Restore full training state from a checkpoint."""
    ckpt = torch.load(path, map_location=config_global.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    rng_path = path.replace(".pt", "_rng.npy")
    if os.path.exists(rng_path):
        rng.set_state(np.load(rng_path, allow_pickle=True).item())
    print(f"🔄 Resumed from checkpoint → {os.path.basename(path)}")


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
    if os.path.exists(src):
        os.rename(src, dst)
    src_rng = src.replace(".pt", "_rng.npy")
    dst_rng = dst.replace(".pt", "_rng.npy")
    if os.path.exists(src_rng):
        os.rename(src_rng, dst_rng)


def _run_epoch(
    model, optimizer, scheduler, scaler,
    datasets: list,
    weights: np.ndarray,
    rng: np.random.RandomState,
    steps_per_epoch: int,
    epoch_desc: str,
) -> float:
    """
    Run one full training epoch (steps_per_epoch gradient updates).
    Returns average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    total_micro_steps = steps_per_epoch * cfg.ACCUMULATION_STEPS
    pbar = tqdm(range(total_micro_steps), desc=epoch_desc, dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    grad_step = 0

    for local_step in pbar:
        x, y = _sample_batch(datasets, cfg.BATCH_SIZE, weights, rng)

        with autocast(enabled=(config_global.DEVICE == "cuda")):
            _, loss = model(x, y)
            loss = loss / cfg.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (local_step + 1) % cfg.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            grad_step += 1

        epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS

        if grad_step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}",
                "lr"  : f"{scheduler.get_last_lr()[0]:.2e}",
            })

    avg_loss = epoch_loss / total_micro_steps
    return avg_loss


# ---------------------------------------------------------------------------
# Curriculum training  (4 sequential stages, blended weights per stage)
# ---------------------------------------------------------------------------

def _train_curriculum(resume: bool) -> None:
    """
    Train sequentially through 4 curriculum stages.

    Each stage runs for cfg.EPOCHS epochs using that stage's blend weights:
        Stage 1 → [0.90, 0.10, 0.00, 0.00]
        Stage 2 → [0.15, 0.75, 0.10, 0.00]
        Stage 3 → [0.05, 0.15, 0.70, 0.10]
        Stage 4 → [0.05, 0.05, 0.15, 0.75]

    Disk footprint during training:
        Only ONE intermediate checkpoint per stage at a time.
        Each epoch's checkpoint deletes the previous epoch's checkpoint.
        When a stage finishes, its checkpoint is renamed *_final.pt and kept.

    Final checkpoints (4 files, one per stage):
        curriculum_stage1_final.pt
        curriculum_stage2_final.pt
        curriculum_stage3_final.pt
        curriculum_stage4_final.pt
    """
    print("🚀 Curriculum Training  |  Sequential stages with blended weights")
    config_global.set_seed()
    cfg.USE_CURRICULUM = True
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    state_file = os.path.join(cfg.CHECKPOINT_DIR, "train_state_curriculum.json")

    datasets = [TokenDataset(cfg.STAGE_BINS[i], cfg.SEQ_LEN) for i in range(1, 5)]

    # Compute total gradient steps across all stages for the LR scheduler
    stage_steps = [
        len(ds) // (cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS)
        for ds in datasets
    ]
    total_steps = sum(stage_steps) * cfg.EPOCHS
    print(f"📐 Total gradient steps (all stages × all epochs): {total_steps:,}")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)
    rng = np.random.RandomState(config_global.RANDOM_SEED)

    # ── resume state ──────────────────────────────────────────────────────
    # state = {"stage": 1-4, "epoch": epochs completed so far in that stage}
    start_stage = 1
    start_epoch = 0  # number of epochs already completed in start_stage

    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        start_stage = state["stage"]
        start_epoch = state["epoch"]
        print(f"🔄 Resuming: Stage {start_stage}, epoch {start_epoch + 1}/{cfg.EPOCHS}")

        ckpt_path = os.path.join(
            cfg.CHECKPOINT_DIR,
            f"curriculum_stage{start_stage}_epoch{start_epoch}.pt",
        )
        if os.path.exists(ckpt_path):
            _load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, rng)
        else:
            print(f"⚠️  Checkpoint not found ({os.path.basename(ckpt_path)}). Starting stage fresh.")
            start_epoch = 0

    # ── main stage loop ───────────────────────────────────────────────────
    for stage in range(start_stage, 5):
        stage_idx = stage - 1
        dataset   = datasets[stage_idx]
        weights   = np.array(cfg.CURRICULUM_STAGE_WEIGHTS[stage], dtype=float)
        steps_ep  = stage_steps[stage_idx]

        print(f"\n{'='*62}")
        print(f"📚 Stage {stage}/4  |  {steps_ep:,} gradient steps/epoch  |  {cfg.EPOCHS} epochs")
        print(f"   Blend weights (S1/S2/S3/S4): {cfg.CURRICULUM_STAGE_WEIGHTS[stage]}")
        print(f"{'='*62}")

        epoch_start = start_epoch if stage == start_stage else 0
        prev_ckpt   = None

        # If resuming mid-stage, the previous epoch checkpoint is already on disk
        if resume and stage == start_stage and start_epoch > 0:
            prev_ckpt = os.path.join(
                cfg.CHECKPOINT_DIR,
                f"curriculum_stage{stage}_epoch{start_epoch}.pt",
            )

        for epoch in range(epoch_start, cfg.EPOCHS):
            desc     = f"Stage {stage}/4 | Epoch {epoch + 1}/{cfg.EPOCHS}"
            avg_loss = _run_epoch(
                model, optimizer, scheduler, scaler,
                datasets, weights, rng, steps_ep, desc,
            )
            print(f"\n   ✅ Stage {stage} | Epoch {epoch + 1}/{cfg.EPOCHS} | avg loss: {avg_loss:.4f}")

            # Save checkpoint for this epoch
            cur_ckpt = os.path.join(
                cfg.CHECKPOINT_DIR,
                f"curriculum_stage{stage}_epoch{epoch + 1}.pt",
            )
            _save_checkpoint(cur_ckpt, model, optimizer, scheduler, scaler, rng)

            # Delete previous epoch's checkpoint (keep disk lean)
            _delete_if_exists(prev_ckpt)
            prev_ckpt = cur_ckpt

            # Persist training state (for crash recovery)
            with open(state_file, "w") as f:
                json.dump({"stage": stage, "epoch": epoch + 1}, f)

        # ── stage complete: promote last epoch file → *_final.pt ──────────
        final_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"curriculum_stage{stage}_final.pt")
        _rename_checkpoint(prev_ckpt, final_ckpt)
        print(f"\n🏆 Stage {stage} complete! Permanent checkpoint → {os.path.basename(final_ckpt)}")

        # Advance state so resume starts at next stage
        with open(state_file, "w") as f:
            json.dump({"stage": stage + 1, "epoch": 0}, f)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n✅ Curriculum training complete!")
    print("📁 Stage checkpoints:")
    for s in range(1, 5):
        fname = f"curriculum_stage{s}_final.pt"
        fpath = os.path.join(cfg.CHECKPOINT_DIR, fname)
        tag   = "✓" if os.path.exists(fpath) else "✗ MISSING"
        print(f"   {tag}  {fname}")


# ---------------------------------------------------------------------------
# Random training  (equal weights, epoch-level crash-recovery checkpoints)
# ---------------------------------------------------------------------------

def _train_random(resume: bool) -> None:
    """
    Train with equal 25% sampling across all 4 stage datasets.

    A checkpoint is written after every epoch; the previous one is deleted,
    so at most ONE checkpoint lives on disk at any time.
    If training crashes mid-epoch, that epoch is lost and training resumes
    from the last fully completed epoch.

    Final checkpoint (1 file):
        random_final.pt
    """
    print("🚀 Random Training  |  Equal 25% weights across all stages")
    config_global.set_seed()
    cfg.USE_CURRICULUM = False
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    state_file = os.path.join(cfg.CHECKPOINT_DIR, "train_state_random.json")

    datasets   = [TokenDataset(cfg.STAGE_BINS[i], cfg.SEQ_LEN) for i in range(1, 5)]
    total_seqs = sum(len(d) for d in datasets)
    steps_ep   = total_seqs // (cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS)
    total_steps = steps_ep * cfg.EPOCHS
    weights    = np.array([0.25, 0.25, 0.25, 0.25])

    print(f"📐 Steps/epoch: {steps_ep:,}  |  Total steps: {total_steps:,}")

    model, optimizer, scheduler, scaler = _build_model_and_optimizer(total_steps)
    rng = np.random.RandomState(config_global.RANDOM_SEED)

    # ── resume state ──────────────────────────────────────────────────────
    start_epoch = 0
    prev_ckpt   = None

    if resume and os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state["epoch"]
        print(f"🔄 Resuming from epoch {start_epoch + 1}/{cfg.EPOCHS}")

        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            _load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, rng)
            prev_ckpt = ckpt_path
        else:
            print(f"⚠️  Checkpoint not found ({os.path.basename(ckpt_path)}). Starting fresh.")
            start_epoch = 0

    # ── main epoch loop ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"🎲 Random training  |  {steps_ep:,} gradient steps/epoch  |  {cfg.EPOCHS} epochs")
    print(f"   Starting from epoch: {start_epoch + 1}")
    print(f"{'='*62}")

    for epoch in range(start_epoch, cfg.EPOCHS):
        desc     = f"Random | Epoch {epoch + 1}/{cfg.EPOCHS}"
        avg_loss = _run_epoch(
            model, optimizer, scheduler, scaler,
            datasets, weights, rng, steps_ep, desc,
        )
        print(f"\n   ✅ Epoch {epoch + 1}/{cfg.EPOCHS} | avg loss: {avg_loss:.4f}")

        # Save checkpoint for this epoch
        cur_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"random_epoch{epoch + 1}.pt")
        _save_checkpoint(cur_ckpt, model, optimizer, scheduler, scaler, rng)

        # Delete previous epoch's checkpoint
        _delete_if_exists(prev_ckpt)
        prev_ckpt = cur_ckpt

        # Persist training state
        with open(state_file, "w") as f:
            json.dump({"epoch": epoch + 1}, f)

    # ── rename last epoch checkpoint → random_final.pt ───────────────────
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