"""
comparator.py
Formats evaluation results into a side-by-side comparison table,
runs statistical tests, and exports results to JSON.
"""
import os
import json
import numpy as np
import Evaluation.config as ec
from src.stats import mcnemar_test, bootstrap_ci_ppl, bootstrap_ci_acc, compute_ece

import sys
sys.stdout = open('logs.txt', 'a')

W    = 80
SEP  = "─" * W
DSEP = "═" * W


def _fmt(v: float) -> str:
    """Format a scalar for the table."""
    if v != v:                 # NaN guard
        return "N/A"
    if v < 1.5:
        return f"{v * 100:.2f}%"
    if v < 1e5:
        return f"{v:.4f}"
    return "N/A"


def _winner(c_val, r_val, lower_better=True) -> str:
    if lower_better:
        return "Curriculum ✓" if c_val < r_val else "Random     ✓"
    return "Curriculum ✓" if c_val > r_val else "Random     ✓"


def _row(metric, c_val, r_val, lower_better=True) -> str:
    return (f"  {metric:<36} {_fmt(c_val):>13}  {_fmt(r_val):>13}  "
            f"{_winner(c_val, r_val, lower_better)}")


def print_comparison_table(curr: dict, rand: dict) -> None:
    """
    Print full comparison table including BPC, new benchmarks,
    bootstrap CIs, McNemar's p-values, and ECE.

    Expected keys per model dict:
        ppl        → {"overall", "bpc", "nll_values"}
        lambada    → {"acc", "correct_mask", "confidences"}
        hellaswag  → {"acc", "correct_mask", "confidences"}
        openbookqa → {"acc", "correct_mask", "confidences"}
        arc_easy   → {"acc", "correct_mask", "confidences"}
        winogrande → {"acc", "correct_mask", "confidences"}
    """

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + DSEP)
    print("  EVALUATION RESULTS — Curriculum vs Random".center(W))
    print(DSEP)
    print(f"  {'Metric':<36} {'Curriculum':>13}  {'Random':>13}  Winner")
    print(SEP)

    # ── Perplexity + BPC ──────────────────────────────────────────────────
    print("  PERPLEXITY on held-out TEST SET  (lower is better)")
    print(SEP)
    print(_row("  Overall PPL (test set)",
               curr["ppl"]["overall"], rand["ppl"]["overall"]))
    print(_row("  Bits-per-Character (BPC)",
               curr["ppl"]["bpc"],     rand["ppl"]["bpc"]))
    print(SEP)

    # ── Accuracy benchmarks ───────────────────────────────────────────────
    benchmarks = [
        ("lambada",    "LAMBADA Accuracy"),
        ("hellaswag",  "HellaSwag Accuracy"),
        ("openbookqa", "OpenBookQA Accuracy"),
        ("arc_easy",   "ARC-Easy Accuracy"),
        ("winogrande", "WinoGrande Accuracy"),
    ]
    print("  BENCHMARKS  (higher is better)")
    print(SEP)
    for key, label in benchmarks:
        print(_row(f"  {label}", curr[key]["acc"], rand[key]["acc"], lower_better=False))
    print(DSEP)

    # ── Statistical Rigor ─────────────────────────────────────────────────
    print("\n" + DSEP)
    print("  STATISTICAL ANALYSIS".center(W))
    print(DSEP)

    # Bootstrap CI for PPL
    print("  Bootstrap 95% Confidence Intervals  (2 000 resamples)")
    print(SEP)
    if curr["ppl"]["nll_values"] and rand["ppl"]["nll_values"]:
        c_lo, c_hi = bootstrap_ci_ppl(curr["ppl"]["nll_values"])
        r_lo, r_hi = bootstrap_ci_ppl(rand["ppl"]["nll_values"])
        print(f"  {'PPL — Curriculum':<36} [{c_lo:.4f}, {c_hi:.4f}]")
        print(f"  {'PPL — Random':<36} [{r_lo:.4f}, {r_hi:.4f}]")
    print(SEP)

    # Bootstrap CI + McNemar's for accuracy benchmarks
    print(f"  {'Benchmark':<22} {'Curr CI':>20}  {'Rand CI':>20}  {'McNemar p':>10}  Sig?")
    print(SEP)
    for key, label in benchmarks:
        c_mask = curr[key]["correct_mask"]
        r_mask = rand[key]["correct_mask"]
        if not c_mask or not r_mask:
            print(f"  {label:<22}  {'N/A':>20}  {'N/A':>20}  {'N/A':>10}")
            continue
        c_lo, c_hi = bootstrap_ci_acc(c_mask)
        r_lo, r_hi = bootstrap_ci_acc(r_mask)
        p = mcnemar_test(c_mask, r_mask)
        sig = "✅ YES" if p < 0.05 else "❌ NO "
        print(f"  {label:<22}  [{c_lo*100:.1f}%, {c_hi*100:.1f}%]"
              f"  [{r_lo*100:.1f}%, {r_hi*100:.1f}%]  {p:>10.4f}  {sig}")
    print(DSEP)

    # ECE
    print("\n" + DSEP)
    print("  CALIBRATION — Expected Calibration Error  (lower = better calibrated)".center(W))
    print(DSEP)
    print(f"  {'Benchmark':<36} {'Curriculum':>13}  {'Random':>13}  Winner")
    print(SEP)
    for key, label in benchmarks:
        c_ece = compute_ece(curr[key]["confidences"], curr[key]["correct_mask"])
        r_ece = compute_ece(rand[key]["confidences"], rand[key]["correct_mask"])
        print(_row(f"  {label}", c_ece, r_ece, lower_better=True))
    print(DSEP)

    # ── Win summary ───────────────────────────────────────────────────────
    ppl_win = 1 if curr["ppl"]["overall"] < rand["ppl"]["overall"] else 0
    acc_wins = sum(1 for k, _ in benchmarks if curr[k]["acc"] > rand[k]["acc"])
    ece_wins = sum(
        1 for k, _ in benchmarks
        if compute_ece(curr[k]["confidences"], curr[k]["correct_mask"]) <
           compute_ece(rand[k]["confidences"], rand[k]["correct_mask"])
    )
    print(f"\n  🏆 Curriculum wins:")
    print(f"      PPL       : {ppl_win}/1")
    print(f"      Accuracy  : {acc_wins}/{len(benchmarks)}")
    print(f"      ECE       : {ece_wins}/{len(benchmarks)}")
    print(DSEP)

    # ── Save JSON (scalars only) ───────────────────────────────────────────
    def _scalar_results(r: dict) -> dict:
        """Strip large lists before saving to JSON."""
        return {
            "ppl":        {"overall": r["ppl"]["overall"], "bpc": r["ppl"]["bpc"]},
            "lambada":    {"acc": r["lambada"]["acc"]},
            "hellaswag":  {"acc": r["hellaswag"]["acc"]},
            "openbookqa":       {"acc": r["openbookqa"]["acc"]},
            "arc_easy":   {"acc": r["arc_easy"]["acc"]},
            "winogrande": {"acc": r["winogrande"]["acc"]},
        }

    out_path = os.path.join(ec.CHECKPOINT_DIR, "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"curriculum": _scalar_results(curr),
                   "random":     _scalar_results(rand)}, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")