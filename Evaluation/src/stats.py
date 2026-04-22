"""
stats.py
Statistical utilities for evaluation rigor.
  - mcnemar_test   : paired significance test for accuracy benchmarks
  - bootstrap_ci   : 95% CI via resampling (works for PPL and accuracy)
  - compute_ece    : Expected Calibration Error for model confidence
"""
import numpy as np
from scipy import stats as scipy_stats


def mcnemar_test(mask_a: list, mask_b: list) -> float:
    """
    McNemar's test (with continuity correction) for two paired boolean arrays.
    mask_a[i] = True  →  model A got example i correct.
    Returns p-value. p < 0.05 means the difference is statistically significant.
    """
    a = np.array(mask_a, dtype=bool)
    b = np.array(mask_b, dtype=bool)
    # b  = A correct, B wrong  |  c = A wrong, B correct
    b_count = int(np.sum(a & ~b))
    c_count = int(np.sum(~a & b))
    if b_count + c_count == 0:
        return 1.0                         # models agree on every example
    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    return float(scipy_stats.chi2.sf(statistic, df=1))


def bootstrap_ci_ppl(nll_values: list, n_bootstrap: int = 2000, ci: float = 0.95):
    """
    Bootstrap confidence interval for perplexity.
    nll_values : per-sequence mean cross-entropy values collected during PPL eval.
    Returns (lower, upper) PPL bounds at the requested confidence level.
    """
    values = np.array(nll_values, dtype=float)
    rng = np.random.default_rng(42)
    boot_ppls = [
        np.exp(min(rng.choice(values, len(values), replace=True).mean(), 10))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_ppls, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def bootstrap_ci_acc(correct_mask: list, n_bootstrap: int = 2000, ci: float = 0.95):
    """
    Bootstrap confidence interval for accuracy.
    correct_mask : list of 1/0 per example.
    Returns (lower, upper) accuracy bounds at the requested confidence level.
    """
    values = np.array(correct_mask, dtype=float)
    rng = np.random.default_rng(42)
    boot_accs = [
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_accs, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def compute_ece(confidences: list, correct_mask: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    confidences  : model's predicted probability for its chosen answer (0-1).
    correct_mask : 1 if the model was right, 0 otherwise.
    Lower ECE = better calibrated model.
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correct_mask, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(conf)
    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = conf[in_bin].mean()
        bin_acc  = corr[in_bin].mean()
        ece += in_bin.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)