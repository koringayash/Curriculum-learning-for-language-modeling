"""
comparator.py
Formats evaluation results into a side-by-side comparison table.
Identifies winners and exports results to JSON.
"""
import os
import json
import Evaluation.config as ec

def print_comparison_table(curr_results: dict, rand_results: dict) -> None:
    """
    Print formatted table comparing Curriculum vs Random models.

    Args:
        curr_results: Dict with 'ppl', 'lambada', 'hellaswag' keys.
                      ppl is now {'overall': float} (test-set perplexity).
        rand_results: Dict with same structure.
    """
    W    = 75
    sep  = "─" * W
    dsep = "═" * W

    print("\n" + dsep)
    print("  EVALUATION RESULTS — Curriculum vs Random".center(W))
    print(dsep)
    print(f"  {'Metric':<35} {'Curriculum':>15}  {'Random':>15}  Winner")
    print(sep)

    def row(metric, c_val, r_val, lower_better=True):
        if lower_better:
            winner = "Curriculum ✓" if c_val < r_val else "Random ✓"
        else:
            winner = "Curriculum ✓" if c_val > r_val else "Random ✓"
        # format: percentage if float < 1.5, else plain float, else N/A
        def fmt(v):
            if isinstance(v, float) and v < 1.5:
                return f"{v * 100:.2f}%"
            if v < 1e5:
                return f"{v:.2f}"
            return "N/A"
        return f"  {metric:<35} {fmt(c_val):>15}  {fmt(r_val):>15}  {winner}"

    # ── ✅ FIX: Single overall test-set perplexity (not per training stage) ─
    print("  PERPLEXITY on held-out TEST SET  (lower is better)")
    print(sep)
    print(row("  Overall (test set)", curr_results["ppl"]["overall"],
              rand_results["ppl"]["overall"]))
    print(sep)

    print("  BENCHMARKS  (higher is better)")
    print(sep)
    print(row("  LAMBADA Accuracy",   curr_results["lambada"],   rand_results["lambada"],   lower_better=False))
    print(row("  HellaSwag Accuracy", curr_results["hellaswag"], rand_results["hellaswag"], lower_better=False))
    print(dsep)

    # Count wins
    ppl_win = 1 if curr_results["ppl"]["overall"] < rand_results["ppl"]["overall"] else 0
    acc_wins = (
        (1 if curr_results["lambada"]   > rand_results["lambada"]   else 0) +
        (1 if curr_results["hellaswag"] > rand_results["hellaswag"] else 0)
    )

    print(f"\n  🏆 Curriculum wins: {ppl_win}/1 perplexity metric | {acc_wins}/2 benchmark metrics")
    print(dsep)

    # Save to JSON
    out_path = os.path.join(ec.CHECKPOINT_DIR, "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"curriculum": curr_results, "random": rand_results}, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")