"""
evaluator.py
Contains all evaluation logic: Perplexity, LAMBADA, and HellaSwag.
Uses streaming datasets and log-prob scoring for zero-shot benchmarks.
"""
import os
import sys
import math
import html
import torch
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import config_global
import Evaluation.config as ec
import Model.config as mc
from Model.src.dataset import TokenDataset
from src.utils import timer  # Reuse timing decorator


def clean_hellaswag_text(text: str) -> str:
    """Strip wiki-style tags and HTML entities from HellaSwag examples."""
    text = html.unescape(text)
    for tag in ["[header]", "[substeps]", "[step]", "[title]"]:
        text = text.replace(tag, "")
    return text.strip()


@torch.no_grad()
def score_completion(model, tokenizer: Tokenizer, context: str, target: str) -> float:
    """
    Compute average log-probability of target tokens given context.
    Used by both LAMBADA and HellaSwag scoring.
    """
    tokenizer.no_padding()
    tokenizer.no_truncation()

    ctx_ids = tokenizer.encode(context).ids
    target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return float("-inf")

    full_ids = ctx_ids + target_ids
    max_len = mc.SEQ_LEN
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        ctx_ids = ctx_ids[overflow:]
        full_ids = ctx_ids + target_ids

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=config_global.DEVICE)
    logits, _ = model(x)

    target_start = len(ctx_ids) - 1
    if target_start < 0 or target_start >= logits.shape[1]:
        return float("-inf")

    log_probs = torch.log_softmax(logits[0], dim=-1)
    total_lp = sum(log_probs[target_start + i, tid].item() for i, tid in enumerate(target_ids) if target_start + i < log_probs.shape[0])
    return total_lp / max(len(target_ids), 1)


@torch.no_grad()
@timer("Perplexity Evaluation")
def run_perplexity_eval(model, model_name: str) -> dict:
    """
    Compute perplexity on overall test set and per-stage binaries.
    Returns dict: {'overall': float, 1: float, 2: float, 3: float, 4: float}
    """
    print(f"\n📈 Perplexity Evaluation — {model_name}")
    results = {}
    total_nll = 0.0
    total_tokens = 0

    for stage_id in range(1, 5):
        bin_path = mc.STAGE_BINS[stage_id]
        if not os.path.exists(bin_path):
            results[stage_id] = float("inf")
            continue

        dataset = TokenDataset(bin_path, mc.SEQ_LEN)
        if len(dataset) == 0:
            results[stage_id] = float("inf")
            continue

        stage_loss = 0.0
        n_batches = 0
        for idx in tqdm(range(len(dataset)), desc=f"  PPL Stage {stage_id}", unit="seq", leave=False):
            x, y = dataset[idx]
            x, y = x.unsqueeze(0).to(config_global.DEVICE), y.unsqueeze(0).to(config_global.DEVICE)
            _, loss = model(x, y)
            stage_loss += loss.item()
            n_batches += 1

        avg_loss = stage_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 10))
        results[stage_id] = ppl

        n_tokens = os.path.getsize(bin_path) // 2
        if n_tokens > 0 and ppl < float("inf"):
            total_nll += math.log(ppl) * n_tokens
            total_tokens += n_tokens
        print(f"    Stage {stage_id}: PPL = {ppl:.2f}")

    results["overall"] = math.exp(min(total_nll / max(total_tokens, 1), 10)) if total_tokens > 0 else float("inf")
    print(f"    Overall: PPL = {results['overall']:.2f}")
    return results


@torch.no_grad()
@timer("LAMBADA Evaluation")
def run_lambada(model, tokenizer: Tokenizer, model_name: str) -> float:
    """
    Zero-shot LAMBADA accuracy: predict last word of passage.
    """
    print(f"\n🎯 LAMBADA Evaluation — {model_name}")
    dataset = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True, trust_remote_code=True)
    tokenizer.no_padding(); tokenizer.no_truncation()

    correct = total = 0
    pbar = tqdm(dataset, desc="  LAMBADA", unit="ex", total=ec.EVAL_MAX_LAMBADA, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_LAMBADA:
            break
        text = example["text"].strip()
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue

        context, last_word = parts
        last_word = last_word.strip().rstrip(".,!?;:")
        if not last_word:
            continue

        ctx_ids = tokenizer.encode(context + " ").ids
        if len(ctx_ids) >= mc.SEQ_LEN:
            ctx_ids = ctx_ids[-(mc.SEQ_LEN - 1):]

        target_first = tokenizer.encode(last_word).ids
        if not target_first:
            continue

        x = torch.tensor([ctx_ids], dtype=torch.long, device=config_global.DEVICE)
        logits, _ = model(x)
        predicted = logits[0, -1, :].argmax().item()

        if predicted == target_first[0]:
            correct += 1
        total += 1
        pbar.set_postfix({"acc": f"{correct/total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc*100:.2f}%")
    return acc


@torch.no_grad()
@timer("HellaSwag Evaluation")
def run_hellaswag(model, tokenizer: Tokenizer, model_name: str) -> float:
    """
    Zero-shot HellaSwag accuracy: pick most probable continuation.
    """
    print(f"\n🧩 HellaSwag Evaluation — {model_name}")
    dataset = load_dataset("Rowan/hellaswag", split="validation", streaming=True, trust_remote_code=True)
    correct = total = 0
    pbar = tqdm(dataset, desc="  HellaSwag", unit="ex", total=ec.EVAL_MAX_HELLASWAG, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_HELLASWAG:
            break
        context = clean_hellaswag_text(example["ctx"])
        endings = [clean_hellaswag_text(e) for e in example["endings"]]
        label = int(example["label"])
        if not context or len(endings) != 4:
            continue

        scores = [score_completion(model, tokenizer, context, end) for end in endings]
        predicted = int(np.argmax(scores))
        if predicted == label:
            correct += 1
        total += 1
        pbar.set_postfix({"acc": f"{correct/total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc*100:.2f}%")
    return acc