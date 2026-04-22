"""
evaluator.py
Contains all evaluation logic: Perplexity, LAMBADA, HellaSwag,
OpenBookQA, ARC-Easy, and WinoGrande.
Uses streaming datasets and log-prob scoring for zero-shot benchmarks.

Return-value contracts
──────────────────────
run_perplexity_eval → {"overall": float, "bpc": float, "nll_values": list[float]}
run_lambada         → {"acc": float, "correct_mask": list[int], "confidences": list[float]}
run_hellaswag       → {"acc": float, "correct_mask": list[int], "confidences": list[float]}
run_openbookqa      → {"acc": float, "correct_mask": list[int], "confidences": list[float]}
run_arc_easy        → {"acc": float, "correct_mask": list[int], "confidences": list[float]}
run_winogrande      → {"acc": float, "correct_mask": list[int], "confidences": list[float]}
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
import Dataset.config as dc
from Model.src.dataset import TokenDataset
from src.utils import timer

sys.stdout = open('logs.txt', 'a')


# ── Helpers ────────────────────────────────────────────────────────────────

def clean_hellaswag_text(text: str) -> str:
    text = html.unescape(text)
    for tag in ["[header]", "[substeps]", "[step]", "[title]"]:
        text = text.replace(tag, "")
    return text.strip()


def _softmax_confidence(scores: list) -> float:
    arr = np.array(scores, dtype=float)
    arr -= arr.max()
    probs = np.exp(arr) / np.exp(arr).sum()
    return float(probs[int(np.argmax(scores))])


@torch.no_grad()
def score_completion(model, tokenizer: Tokenizer, context: str, target: str) -> float:
    tokenizer.no_padding()
    tokenizer.no_truncation()

    ctx_ids    = tokenizer.encode(context).ids
    target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return float("-inf")

    full_ids = ctx_ids + target_ids
    max_len  = mc.SEQ_LEN
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        ctx_ids  = ctx_ids[overflow:]
        full_ids = ctx_ids + target_ids

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=config_global.DEVICE)
    logits, _ = model(x)

    target_start = len(ctx_ids) - 1
    if target_start < 0 or target_start >= logits.shape[1]:
        return float("-inf")

    log_probs = torch.log_softmax(logits[0], dim=-1)
    total_lp  = sum(
        log_probs[target_start + i, tid].item()
        for i, tid in enumerate(target_ids)
        if target_start + i < log_probs.shape[0]
    )
    return total_lp / max(len(target_ids), 1)


# ── Core evaluators ────────────────────────────────────────────────────────

@torch.no_grad()
@timer("Perplexity Evaluation")
def run_perplexity_eval(model, model_name: str) -> dict:
    print(f"\n📈 Perplexity Evaluation (test set) — {model_name}")

    test_bin = dc.TEST_TOKENS
    if not os.path.exists(test_bin):
        raise FileNotFoundError(
            f"❌ Test corpus not found at {test_bin}.\n"
            "   Run Dataset/run.py first."
        )

    dataset = TokenDataset(test_bin, mc.SEQ_LEN)
    if len(dataset) == 0:
        print("⚠️  Test dataset is empty — returning inf.")
        return {"overall": float("inf"), "bpc": float("inf"), "nll_values": []}

    total_nll    = 0.0
    total_tokens = 0
    nll_values   = []

    for idx in tqdm(range(len(dataset)), desc="  PPL (test)", unit="seq", leave=False):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(config_global.DEVICE)
        y = y.unsqueeze(0).to(config_global.DEVICE)
        _, loss = model(x, y)
        seq_nll = loss.item()
        nll_values.append(seq_nll)
        total_nll    += seq_nll * mc.SEQ_LEN
        total_tokens += mc.SEQ_LEN

    avg_nll = total_nll / max(total_tokens, 1)
    ppl     = math.exp(min(avg_nll, 10))
    bpc     = avg_nll / math.log(2)

    print(f"    Test PPL = {ppl:.2f} | BPC = {bpc:.4f}  ({len(dataset):,} sequences)")
    return {"overall": ppl, "bpc": bpc, "nll_values": nll_values}


@torch.no_grad()
@timer("LAMBADA Evaluation")
def run_lambada(model, tokenizer: Tokenizer, model_name: str) -> dict:
    print(f"\n🎯 LAMBADA Evaluation — {model_name}")
    dataset = load_dataset(          # trust_remote_code removed
        "EleutherAI/lambada_openai", split="test", streaming=True,
    )
    tokenizer.no_padding()
    tokenizer.no_truncation()

    correct = total = 0
    correct_mask = []
    confidences  = []
    pbar = tqdm(dataset, desc="  LAMBADA", unit="ex",
                total=ec.EVAL_MAX_LAMBADA, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_LAMBADA:
            break
        text  = example["text"].strip()
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

        x           = torch.tensor([ctx_ids], dtype=torch.long, device=config_global.DEVICE)
        logits, _   = model(x)
        last_logits = logits[0, -1, :]
        probs       = torch.softmax(last_logits, dim=-1)
        predicted   = last_logits.argmax().item()

        is_correct = int(predicted == target_first[0])
        correct      += is_correct
        correct_mask.append(is_correct)
        confidences.append(probs[predicted].item())
        total += 1
        pbar.set_postfix({"acc": f"{correct / total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    return {"acc": acc, "correct_mask": correct_mask, "confidences": confidences}


@torch.no_grad()
@timer("HellaSwag Evaluation")
def run_hellaswag(model, tokenizer: Tokenizer, model_name: str) -> dict:
    print(f"\n🧩 HellaSwag Evaluation — {model_name}")
    dataset = load_dataset(          # trust_remote_code removed
        "Rowan/hellaswag", split="validation", streaming=True,
    )
    correct = total = 0
    correct_mask = []
    confidences  = []
    pbar = tqdm(dataset, desc="  HellaSwag", unit="ex",
                total=ec.EVAL_MAX_HELLASWAG, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_HELLASWAG:
            break
        context = clean_hellaswag_text(example["ctx"])
        endings = [clean_hellaswag_text(e) for e in example["endings"]]
        label   = int(example["label"])
        if not context or len(endings) != 4:
            continue

        scores    = [score_completion(model, tokenizer, context, end) for end in endings]
        predicted = int(np.argmax(scores))
        is_correct = int(predicted == label)
        correct      += is_correct
        correct_mask.append(is_correct)
        confidences.append(_softmax_confidence(scores))
        total += 1
        pbar.set_postfix({"acc": f"{correct / total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    return {"acc": acc, "correct_mask": correct_mask, "confidences": confidences}


# ── New benchmarks ─────────────────────────────────────────────────────────

@torch.no_grad()
@timer("OpenBookQA Evaluation")
def run_openbookqa(model, tokenizer: Tokenizer, model_name: str) -> dict:
    """
    Zero-shot OpenBookQA: elementary science questions (4-choice).
    Replaces PIQA (ybisk/piqa uses a legacy loading script incompatible
    with datasets >= 3.x).
    Dataset : allenai/openbookqa  |  split: test
    Fields  : question_stem, choices {"text", "label"}, answerKey ("A"-"D")
    """
    print(f"\n📖 OpenBookQA Evaluation — {model_name}")
    dataset = load_dataset(
        "allenai/openbookqa", "main", split="test", streaming=True,
    )
    correct = total = 0
    correct_mask = []
    confidences  = []
    key_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    pbar = tqdm(dataset, desc="  OpenBookQA", unit="ex",
                total=ec.EVAL_MAX_OPENBOOKQA, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_OPENBOOKQA:
            break
        question = example["question_stem"].strip()
        choices  = example["choices"]["text"]
        ans_key  = example["answerKey"].strip()
        if ans_key not in key_map or not choices:
            continue
        label = key_map[ans_key]
        if label >= len(choices):
            continue

        scores    = [score_completion(model, tokenizer, question, c) for c in choices]
        predicted = int(np.argmax(scores))
        is_correct = int(predicted == label)
        correct      += is_correct
        correct_mask.append(is_correct)
        confidences.append(_softmax_confidence(scores))
        total += 1
        pbar.set_postfix({"acc": f"{correct / total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    return {"acc": acc, "correct_mask": correct_mask, "confidences": confidences}


@torch.no_grad()
@timer("ARC-Easy Evaluation")
def run_arc_easy(model, tokenizer: Tokenizer, model_name: str) -> dict:
    print(f"\n🔬 ARC-Easy Evaluation — {model_name}")
    dataset = load_dataset(          # trust_remote_code removed
        "allenai/ai2_arc", "ARC-Easy", split="test", streaming=True,
    )
    correct = total = 0
    correct_mask = []
    confidences  = []
    key_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
               "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    pbar = tqdm(dataset, desc="  ARC-Easy", unit="ex",
                total=ec.EVAL_MAX_ARC_EASY, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_ARC_EASY:
            break
        question = example["question"].strip()
        choices  = example["choices"]["text"]
        ans_key  = example["answerKey"].strip()
        if ans_key not in key_map or not choices:
            continue
        label = key_map[ans_key]
        if label >= len(choices):
            continue

        scores    = [score_completion(model, tokenizer, question, c) for c in choices]
        predicted = int(np.argmax(scores))
        is_correct = int(predicted == label)
        correct      += is_correct
        correct_mask.append(is_correct)
        confidences.append(_softmax_confidence(scores))
        total += 1
        pbar.set_postfix({"acc": f"{correct / total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    return {"acc": acc, "correct_mask": correct_mask, "confidences": confidences}


@torch.no_grad()
@timer("WinoGrande Evaluation")
def run_winogrande(model, tokenizer: Tokenizer, model_name: str) -> dict:
    print(f"\n🧠 WinoGrande Evaluation — {model_name}")
    dataset = load_dataset(          # trust_remote_code removed
        "allenai/winogrande", "winogrande_xl", split="validation", streaming=True,
    )
    correct = total = 0
    correct_mask = []
    confidences  = []
    pbar = tqdm(dataset, desc="  WinoGrande", unit="ex",
                total=ec.EVAL_MAX_WINOGRANDE, leave=False)

    for example in pbar:
        if total >= ec.EVAL_MAX_WINOGRANDE:
            break
        sentence = example["sentence"]
        options  = [example["option1"].strip(), example["option2"].strip()]
        answer   = example["answer"].strip()
        if "_" not in sentence or answer not in ("1", "2"):
            continue
        label = int(answer) - 1

        blank_idx = sentence.index("_")
        prefix    = sentence[:blank_idx]
        suffix    = sentence[blank_idx + 1:]

        scores    = [score_completion(model, tokenizer, prefix, opt + suffix)
                     for opt in options]
        predicted = int(np.argmax(scores))
        is_correct = int(predicted == label)
        correct      += is_correct
        correct_mask.append(is_correct)
        confidences.append(_softmax_confidence(scores))
        total += 1
        pbar.set_postfix({"acc": f"{correct / total:.3f}"})

    acc = correct / max(total, 1)
    print(f"    Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    return {"acc": acc, "correct_mask": correct_mask, "confidences": confidences}