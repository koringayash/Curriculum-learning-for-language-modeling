"""curriculum_sampler.py: Hybrid curriculum learning batch sampling."""
import numpy as np
import torch
import Model.config as cfg
import config_global

def get_curriculum_weights(progress: float) -> list:
    boundaries = [0.0, 0.25, 0.50, 0.75, 1.01]
    current_stage = next(i+1 for i in range(4) if progress < boundaries[i+1])
    w_current = cfg.CURRICULUM_STAGE_WEIGHTS[current_stage]
    if current_stage < 4:
        stage_boundary = 0.25 * current_stage
        blend_start = stage_boundary - cfg.CURRICULUM_BLEND_FRACTION
        if progress >= blend_start:
            alpha = min((progress - blend_start) / cfg.CURRICULUM_BLEND_FRACTION, 1.0)
            w_next = cfg.CURRICULUM_STAGE_WEIGHTS[current_stage + 1]
            return [w_current[j]*(1-alpha) + w_next[j]*alpha for j in range(4)]
    return list(w_current)

def get_batch(datasets: list, batch_size: int, progress: float, rng: np.random.RandomState) -> tuple:
    weights = np.array(get_curriculum_weights(progress) if cfg.USE_CURRICULUM else [0.25]*4)
    weights /= weights.sum()
    xs, ys = [], []
    for _ in range(batch_size):
        stage_idx = rng.choice(4, p=weights)
        ds = datasets[stage_idx]
        sample_idx = rng.randint(0, len(ds))
        x, y = ds[sample_idx]
        xs.append(x); ys.append(y)
    return torch.stack(xs).to(config_global.DEVICE), torch.stack(ys).to(config_global.DEVICE)

def describe_weights(progress: float) -> str:
    w = get_curriculum_weights(progress) if cfg.USE_CURRICULUM else [0.25]*4
    return f"w=[{int(w[0]*100)}/{int(w[1]*100)}/{int(w[2]*100)}/{int(w[3]*100)}]"