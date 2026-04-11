"""
comparator.py
Formats evaluation results into a side-by-side comparison table.
Identifies winners and exports results to JSON.
"""
import os
import json
import Evaluation.config as ec

STAGE_NAMES = {
    1: "Easy       (Stage 1)",
    2: "Medium     (Stage 2)",
    3: "Hard       (Stage 3)",
    4: "Very Hard  (Stage 4)",
}


def print_comparison_table(curr_results: dict, rand_results: dict) -> None:
    """
    Print formatted table comparing Curriculum vs Random models.

    Args:
        curr_results: Dict with 'ppl', 'lambada', 'hellaswag' keys.
        rand_results: Dict with same structure.
    """
    W = 75
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
        c_str = f"{c_val*100:.2f}%" if isinstance(c_val, float) and c_val < 1.5 else f"{c_val:.2f}" if c_val < 1e5 else "N/A"
        r_str = f"{r_val*100:.2f}%" if isinstance(r_val, float) and r_val < 1.5 else f"{r_val:.2f}" if r_val < 1e5 else "N/A"
        return f"  {metric:<35} {c_str:>15}  {r_str:>15}  {winner}"

    print("  PERPLEXITY  (lower is better)")
    print(sep)
    print(row("  Overall", curr_results["ppl"]["overall"], rand_results["ppl"]["overall"]))
    for i in range(1, 5):
        print(row(f"  {STAGE_NAMES[i]}", curr_results["ppl"][i], rand_results["ppl"][i]))
    print(sep)

    print("  BENCHMARKS  (higher is better)")
    print(sep)
    print(row("  LAMBADA Accuracy", curr_results["lambada"], rand_results["lambada"], lower_better=False))
    print(row("  HellaSwag Accuracy", curr_results["hellaswag"], rand_results["hellaswag"], lower_better=False))
    print(dsep)

    # Count wins
    ppl_wins = sum(1 for k in ["overall", 1, 2, 3, 4] if curr_results["ppl"][k] < rand_results["ppl"][k])
    acc_wins = (1 if curr_results["lambada"] > rand_results["lambada"] else 0) + \
               (1 if curr_results["hellaswag"] > rand_results["hellaswag"] else 0)

    print(f"\n  🏆 Curriculum wins: {ppl_wins}/5 perplexity metrics | {acc_wins}/2 benchmark metrics")
    print(dsep)

    # Save to JSON
    out_path = os.path.join(mc.CHECKPOINT_DIR, "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"curriculum": curr_results, "random": rand_results}, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")