"""
Evaluation/run.py
Step 7: Compares Curriculum vs Random trained models.
Checks checkpoint existence, runs all evaluations, prints comparison table.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global
import Model.config as mc
from Evaluation.src.loader import find_latest_checkpoint, load_model, load_tokenizer
from Evaluation.src.evaluator import (run_perplexity_eval, run_lambada, run_hellaswag, run_openbookqa, run_arc_easy, run_winogrande)
from Evaluation.src.comparator import print_comparison_table


sys.stdout = open('logs.txt', 'a')

def main():
    print("=" * 60)
    print("Model Evaluation & Comparison")
    print("=" * 60)
    t_start = time.time()

    # 1. Verify checkpoints
    curr_ckpt = find_latest_checkpoint("curriculum")
    rand_ckpt = find_latest_checkpoint("random")
    if not curr_ckpt:
        print("❌ ERROR: No curriculum checkpoint found.")
        print("   → Run: python Model/run.py --mode curriculum")
        sys.exit(1)
    if not rand_ckpt:
        print("❌ ERROR: No random checkpoint found.")
        print("   → Run: python Model/run.py --mode random")
        sys.exit(1)

    print(f"  📂 Curriculum: {os.path.basename(curr_ckpt)}")
    print(f"  📂 Random    : {os.path.basename(rand_ckpt)}")
    print()

    # 2. Load shared resources
    tokenizer = load_tokenizer()
    tokenizer.no_padding()
    tokenizer.no_truncation()

    results = {}
    for mode, ckpt_path in [("curriculum", curr_ckpt), ("random", rand_ckpt)]:
        model = load_model(ckpt_path)
        def timed_eval(name, fn):
            t = time.time()
            result = fn()
            print(f"   ⏱️ {name}: {time.time() - t:.2f}s")
            return result

        results[mode] = {
            "ppl":       timed_eval("Perplexity", lambda: run_perplexity_eval(model, mode)),
            "lambada":   timed_eval("LAMBADA",    lambda: run_lambada(model, tokenizer, mode)),
            "hellaswag": timed_eval("HellaSwag",  lambda: run_hellaswag(model, tokenizer, mode)),
            # ↓ ADD THESE THREE
            "openbookqa": timed_eval("OpenBookQA", lambda: run_openbookqa(model, tokenizer, mode)),
            "arc_easy":   timed_eval("ARC-Easy",   lambda: run_arc_easy(model, tokenizer, mode)),
            "winogrande": timed_eval("WinoGrande", lambda: run_winogrande(model, tokenizer, mode)),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Compare & Print
    print_comparison_table(results["curriculum"], results["random"])
    print(f"\n⏱️  Total Evaluation Time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()