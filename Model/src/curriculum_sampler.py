"""curriculum_sampler.py: Hybrid curriculum learning batch sampling.

Key fix: replaced rng.randint (with-replacement) with a permutation-based
sampler that guarantees every document in every stage is seen before any
document repeats. The weight-blending logic is completely unchanged.
"""
import numpy as np
import torch
import Model.config as cfg
import config_global

import sys

sys.stdout = open('logs.txt', 'a')
# ---------------------------------------------------------------------------
# Weight schedule  (unchanged from original)
# ---------------------------------------------------------------------------

def get_curriculum_weights(progress: float) -> list:
    boundaries = [0.0, 0.25, 0.50, 0.75, 1.01]
    current_stage = next(i + 1 for i in range(4) if progress < boundaries[i + 1])
    w_current = cfg.CURRICULUM_STAGE_WEIGHTS[current_stage]
    if current_stage < 4:
        stage_boundary = 0.25 * current_stage
        blend_start = stage_boundary - cfg.CURRICULUM_BLEND_FRACTION
        if progress >= blend_start:
            alpha = min((progress - blend_start) / cfg.CURRICULUM_BLEND_FRACTION, 1.0)
            w_next = cfg.CURRICULUM_STAGE_WEIGHTS[current_stage + 1]
            return [w_current[j] * (1 - alpha) + w_next[j] * alpha for j in range(4)]
    return list(w_current)


def describe_weights(progress: float) -> str:
    w = get_curriculum_weights(progress) if cfg.USE_CURRICULUM else [0.25] * 4
    return f"w=[{int(w[0]*100)}/{int(w[1]*100)}/{int(w[2]*100)}/{int(w[3]*100)}]"


# ---------------------------------------------------------------------------
# ✅ FIX: Stateful without-replacement sampler
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """
    Maintains one shuffled permutation per stage.
    When a stage's permutation is exhausted it reshuffles and restarts —
    so every document is guaranteed to appear before any document repeats.

    Drop-in replacement for the old get_batch() function.
    Usage:
        sampler = CurriculumSampler(datasets, rng)
        x, y = sampler.get_batch(batch_size, progress)
    """

    def __init__(self, datasets: list, rng: np.random.RandomState):
        self.datasets = datasets
        self.rng = rng
        # One independent shuffled permutation per stage
        self.perms = [rng.permutation(len(d)) for d in datasets]
        self.ptrs  = [0] * 4

    def _next_from_stage(self, stage_idx: int):
        """Return one (x, y) pair from stage_idx without replacement."""
        ptr  = self.ptrs[stage_idx]
        perm = self.perms[stage_idx]

        # Exhausted this stage's permutation → reshuffle and restart
        if ptr >= len(perm):
            self.perms[stage_idx] = self.rng.permutation(
                len(self.datasets[stage_idx])
            )
            self.ptrs[stage_idx] = 0
            ptr = 0

        doc_idx = self.perms[stage_idx][ptr]
        self.ptrs[stage_idx] += 1
        return self.datasets[stage_idx][doc_idx]

    def get_batch(self, batch_size: int, progress: float) -> tuple:
        """
        Build one batch of (x, y) tensors.

        Uses curriculum weights when cfg.USE_CURRICULUM is True,
        equal 25% weights when False — so the same class works for
        both training modes.
        """
        if cfg.USE_CURRICULUM:
            weights = np.array(get_curriculum_weights(progress))
        else:
            weights = np.array([0.25, 0.25, 0.25, 0.25])
        weights /= weights.sum()

        xs, ys = [], []
        for _ in range(batch_size):
            stage_idx = self.rng.choice(4, p=weights)
            x, y = self._next_from_stage(stage_idx)
            xs.append(x)
            ys.append(y)

        return (
            torch.stack(xs).to(config_global.DEVICE),
            torch.stack(ys).to(config_global.DEVICE),
        )