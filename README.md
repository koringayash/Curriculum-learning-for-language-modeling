# 1. Setup
docker build -t nlp-curriculum .
docker run -it --gpus all -v $(pwd):/app nlp-curriculum bash

# 2. Dataset (Steps 1-5)
python Dataset/run.py

# 3. Train Curriculum
python Model/run.py --mode curriculum

# 4. Train Random (Baseline)
python Model/run.py --mode random

# Resume if interrupted:
python Model/run.py --mode curriculum --resume

# 5. Compare
python Evaluation/run.py

# Curriculum Learning for GPT-style Language Models

> Does training order matter? This project trains two identical GPT models — one with a difficulty-staged curriculum, one with random ordering — and rigorously compares them across six NLP benchmarks.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Pipeline: How It Works](#pipeline-how-it-works)
5. [Quickstart](#quickstart)
6. [Configuration Reference](#configuration-reference)
7. [Dataset & Model Statistics](#dataset--model-statistics)
8. [Scoring & Curriculum Design](#scoring--curriculum-design)
9. [Evaluation Benchmarks](#evaluation-benchmarks)
10. [Results](#results)
11. [Statistical Analysis](#statistical-analysis)
12. [Reproducibility](#reproducibility)
13. [Requirements](#requirements)

---

## Overview

This project investigates whether **curriculum learning** — training a language model on easy text first, then progressively harder text — leads to better downstream performance compared to **random training order** (the standard approach).

A GPT-style transformer (~108M parameters) is trained from scratch on 1.5 million documents from the FineWeb web corpus. Each document is assigned a **composite difficulty score** based on perplexity, readability, vocabulary richness, and compression ratio, then sorted into four difficulty stages. Two models are trained under identical conditions, differing only in whether documents are presented in curriculum order or random order. The models are then evaluated head-to-head on six NLP benchmarks.

---

## Architecture

The model is a modern GPT-style decoder-only transformer with the following components:

| Component | Choice | Reason |
|---|---|---|
| Normalization | **RMSNorm** | More stable and efficient than LayerNorm |
| Positional Encoding | **RoPE** (Rotary Positional Embedding) | Better length generalization than learned absolute positions |
| Activation | **SwiGLU** | Empirically stronger than ReLU/GELU in feed-forward blocks |
| Attention | **FlashAttention** | Memory-efficient scaled dot-product attention |
| Embedding Tying | **Weight Tying** | Input and output embeddings are shared, reducing parameters |

**Model hyperparameters:**

| Parameter | Value |
|---|---|
| Sequence Length (`SEQ_LEN`) | 512 |
| Embedding Dimension (`N_EMBD`) | 768 |
| Attention Heads (`N_HEAD`) | 8 |
| Transformer Layers (`N_LAYER`) | 12 |
| Dropout | 0.1 |
| Vocabulary Size | 30,000 |
| **Total Parameters** | **107,993,856 (~108M)** |

---

## Project Structure

```
Final_project/
│
├── config_global.py          # Global config: device detection, seeds, project root
├── Dockerfile                # Containerized environment (Python 3.10-slim + GPU support)
├── requirements.txt          # Python dependencies
├── logs.txt                  # Evaluation output log
│
├── Dataset/                  # Stage 1–5: Download → Tokenize → Encode → Score → Stage
│   ├── config.py             # Dataset-specific hyperparameters and paths
│   ├── run.py                # Orchestrates all 5 dataset preparation steps
│   └── src/
│       ├── downloader.py         # Streams documents from HuggingFace FineWeb
│       ├── tokenizer.py          # Trains a BPE tokenizer from scratch
│       ├── encoder.py            # Encodes documents to binary token files
│       ├── reference_trainer.py  # Trains a small reference GPT to score perplexity
│       └── scorer_stager.py      # Scores docs and splits into 4 difficulty bins
│
├── Model/                    # Training: curriculum vs random mode
│   ├── config.py             # Architecture and training hyperparameters
│   ├── run.py                # Entry point: --mode curriculum | random | --resume
│   └── src/
│       ├── model.py              # GPT model definition (RMSNorm, RoPE, SwiGLU)
│       ├── trainer.py            # Training loop with checkpointing and logging
│       ├── curriculum_sampler.py # Samples documents by stage according to schedule
│       └── dataset.py            # PyTorch dataset wrapper for binary token files
│
└── Evaluation/               # Benchmarking and comparison
    ├── config.py             # Evaluation limits and settings
    ├── run.py                # Loads both checkpoints and runs all benchmarks
    ├── generator.py          # Text generation utilities
    └── src/
        ├── evaluator.py          # All 6 benchmark implementations
        ├── comparator.py         # Side-by-side comparison and statistical tests
        ├── stats.py              # Bootstrap CI, McNemar test, ECE computation
        ├── loader.py             # Checkpoint and model loading utilities
        └── utils.py              # Shared helpers
```

---

## Pipeline: How It Works

The full experiment runs in three sequential phases.

### Phase 1 — Dataset Preparation (`Dataset/run.py`)

| Step | Script | Description |
|---|---|---|
| 1 | `downloader.py` | Streams 1,500,000 train + 5,000 test documents from HuggingFace FineWeb (`CC-MAIN-2024-10`), filtering to documents ≥ 300 characters |
| 2 | `tokenizer.py` | Trains a BPE tokenizer (vocab size 30,000) from scratch on the downloaded corpus |
| 3 | `encoder.py` | Tokenizes all documents and writes them to memory-mapped binary files (`train_tokens.bin`, `test_tokens.bin`, `doc_offsets.json`) |
| 4 | `reference_trainer.py` | Trains a small 2-layer reference GPT (128-dim, 4-head) for 1 epoch, used only for perplexity scoring |
| 5 | `scorer_stager.py` | Scores every document with a composite difficulty metric and writes four stage binary files |

### Phase 2 — Training (`Model/run.py`)

Two models are trained independently with the same architecture, hyperparameters, and random seed — the only difference is document ordering:

- **Curriculum mode**: Documents are served according to `CURRICULUM_EPOCH_STAGES`, which controls which difficulty stages are active each epoch.
- **Random mode**: All documents are shuffled randomly regardless of difficulty.

Checkpoints are saved at 25%, 50%, 75%, and 100% of training for curriculum, and once per epoch for random.

### Phase 3 — Evaluation (`Evaluation/run.py`)

Both final checkpoints (`curriculum_final.pt`, `random_final.pt`) are loaded and evaluated on six benchmarks. Results are compared with bootstrap confidence intervals, McNemar's test, and Expected Calibration Error (ECE).

---

## Quickstart

### Prerequisites
- Docker with GPU support (`nvidia-container-toolkit`)
- At least **[PLACEHOLDER: ~X GB]** of disk space for dataset and checkpoints
- CUDA-capable GPU recommended (CPU and Apple MPS are also supported)

### 1. Build the Docker image

```bash
docker build -t nlp-curriculum .
```

### 2. Start the container

```bash
docker run -it --gpus all -v $(pwd):/app nlp-curriculum bash
```

### 3. Prepare the dataset (Steps 1–5, runs sequentially)

```bash
python Dataset/run.py
```

> ⚠️ This step downloads ~1.5M documents and may take **[PLACEHOLDER: ~X hours]** depending on bandwidth and hardware.

### 4. Train the curriculum model

```bash
python Model/run.py --mode curriculum
```

### 5. Train the random baseline

```bash
python Model/run.py --mode random
```

**To resume an interrupted run:**

```bash
python Model/run.py --mode curriculum --resume
python Model/run.py --mode random --resume
```

### 6. Run evaluation

```bash
python Evaluation/run.py
```

Results are printed to `logs.txt` and saved to `checkpoints/eval_results.json`.

---

## Configuration Reference

Key knobs across the three config files. Edit these to scale or adjust the experiment.

### `Dataset/config.py`

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_DOCS` | `1,500,000` | Number of training documents to download |
| `TEST_DOCS` | `5,000` | Number of test documents |
| `MIN_DOC_CHARS` | `300` | Minimum character length to accept a document |
| `FINEWEB_DATASET` | `HuggingFaceFW/fineweb` | HuggingFace dataset identifier |
| `FINEWEB_CONFIG` | `CC-MAIN-2024-10` | Specific crawl snapshot |
| `VOCAB_SIZE` | `30,000` | BPE tokenizer vocabulary size |
| `SCORE_WEIGHT_PPL` | `0.50` | Perplexity contribution to difficulty score |
| `SCORE_WEIGHT_FLESCH` | `0.30` | Flesch readability contribution |
| `SCORE_WEIGHT_TTR` | `0.10` | Type-Token Ratio contribution |
| `SCORE_WEIGHT_COMP` | `0.10` | Compression ratio contribution |

### `Model/config.py`

| Parameter | Default | Description |
|---|---|---|
| `SEQ_LEN` | `512` | Tokens per training sequence |
| `N_EMBD` | `768` | Embedding dimension |
| `N_HEAD` | `8` | Number of attention heads |
| `N_LAYER` | `12` | Number of transformer layers |
| `DROPOUT` | `0.1` | Dropout probability |
| `BATCH_SIZE` | `8` | Micro-batch size |
| `ACCUMULATION_STEPS` | `4` | Gradient accumulation (effective batch = 32) |
| `LEARNING_RATE` | `2.5e-4` | Peak learning rate |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `EPOCHS` | `1` | Training epochs |
| `GRAD_CLIP` | `1.0` | Gradient clipping norm |
| `CURRICULUM_EPOCH_STAGES` | `{1: [1,2,3,4]}` | Which difficulty stages are active per epoch |

### `Evaluation/config.py`

| Parameter | Default | Description |
|---|---|---|
| `EVAL_MAX_LAMBADA` | `5,000` | Max LAMBADA examples to evaluate |
| `EVAL_MAX_HELLASWAG` | `10,000` | Max HellaSwag examples |
| `EVAL_MAX_OPENBOOKQA` | `500` | Full OpenBookQA validation set |
| `EVAL_MAX_ARC_EASY` | `1,172` | Full ARC-Easy test set |
| `EVAL_MAX_WINOGRANDE` | `1,267` | Full WinoGrande validation set |
| `EVAL_SEQ_LEN` | `512` | Must match or exceed training `SEQ_LEN` |

---

## Dataset & Model Statistics

| Statistic | Value |
|---|---|
| **Source Dataset** | HuggingFace FineWeb (`CC-MAIN-2024-10`) |
| **Training Documents** | 1,500,000 |
| **Test Documents** | 5,000 |
| **Minimum Document Length** | 300 characters |
| **Total Training Tokens** | [PLACEHOLDER: ~X billion tokens] |
| **Total Test Sequences** | 7,197 (512-token windows) |
| **Tokenizer Type** | BPE (trained from scratch) |
| **Vocabulary Size** | 30,000 |
| **Model Parameters** | 107,993,856 (~108M) |
| **Effective Batch Size** | 32 (8 micro × 4 accumulation steps) |
| **Training Epochs** | 1 |
| **Training Steps (curriculum)** | [PLACEHOLDER: ~X steps] |
| **Approx. Training Time (curriculum)** | [PLACEHOLDER: ~X hours on X GPU] |
| **Approx. Training Time (random)** | [PLACEHOLDER: ~X hours on X GPU] |
| **Checkpoint Size** | [PLACEHOLDER: ~X MB per checkpoint] |

---

## Scoring & Curriculum Design

Each training document is assigned a single composite difficulty score that combines four signals:

```
difficulty = 0.50 × PPL_score
           + 0.30 × Flesch_score
           + 0.10 × TTR_score
           + 0.10 × Compression_score
```

| Signal | Weight | What it measures |
|---|---|---|
| **Perplexity (PPL)** | 50% | How surprised the reference model is by the text — higher = harder, less common language patterns |
| **Flesch Reading Ease** | 30% | Sentence length and syllable count — lower score = harder to read |
| **Type-Token Ratio (TTR)** | 10% | Vocabulary richness — higher TTR = more diverse vocabulary = harder |
| **Compression Ratio** | 10% | How much a document compresses — low compression = high information density = harder |

Documents are then split into four equal-sized stages:

| Stage | Label | Description |
|---|---|---|
| 1 | Easy | Simple language, short sentences, common vocabulary |
| 2 | Medium | Moderate complexity, some domain-specific terms |
| 3 | Hard | Complex sentence structures, richer vocabulary |
| 4 | Very Hard | Dense, technical, or highly varied text |

The `CURRICULUM_EPOCH_STAGES` dictionary controls which stages are active during each training epoch. In the current configuration with `EPOCHS=1`, all four stages are used together (effectively approximating the full curriculum in one pass). For a multi-epoch curriculum, a typical schedule would be:

```python
CURRICULUM_EPOCH_STAGES = {
    1: [1],          # Epoch 1: easy only
    2: [1, 2],       # Epoch 2: easy + medium
    3: [1, 2, 3, 4], # Epoch 3: full difficulty range
}
```

---

## Evaluation Benchmarks

All benchmarks are evaluated **zero-shot** using log-probability scoring — the model is never fine-tuned on any of them.

### 1. Perplexity (Test Set)
Measures how well the model predicts the held-out test documents from the same FineWeb distribution. Lower is better. This is the most direct measure of language modeling quality on in-distribution data. Both models achieve similar perplexity (~28.7), confirming they learned comparably from the same corpus.

### 2. LAMBADA
Tests whether the model can predict the final word of a paragraph whose correct answer requires understanding the **entire preceding passage** — not just the last sentence. This demands strong long-range contextual reasoning. Both models score near 0%, which is expected for a 108M-parameter model trained for 1 epoch. LAMBADA typically requires hundreds of millions of parameters and much more training data before meaningful accuracy emerges.

### 3. HellaSwag
A 4-way multiple-choice benchmark where the model must pick the most plausible continuation of a short activity description. Random chance is 25%. Both models score ~27.5%, just above random. At this model scale and training budget, the model has learned surface-level language statistics but lacks the grounded commonsense needed for HellaSwag, which was specifically designed to be hard for models that rely on shallow linguistic cues.

### 4. OpenBookQA
A 4-way multiple-choice science QA benchmark that requires combining an elementary science fact with broader commonsense knowledge (random = 25%). The **random model wins here** (26.60% vs 24.40%). This is plausible — curriculum ordering based on text readability/complexity may cause the model to focus more on stylistic and structural patterns than on factual content, slightly reducing its knowledge-retrieval capability compared to a uniformly shuffled training set.

### 5. ARC-Easy
Multiple-choice science exam questions drawn from US grade school tests (4-way, random = 25%). The **random model also wins** (34.98% vs 34.04%). Like OpenBookQA, fact-based knowledge recall does not appear to benefit from the curriculum ordering used here. The difficulty scoring (perplexity + readability) does not capture factual knowledge density, so the curriculum stages do not align well with what these knowledge-retrieval benchmarks test.

### 6. WinoGrande
A binary-choice pronoun resolution task requiring commonsense reasoning to resolve ambiguous pronouns in sentences (random = 50%). The **curriculum model wins** (51.93% vs 51.07%). This is intuitive — starting with simpler, well-structured text may help the model learn basic entity tracking and co-reference earlier in training, giving it a slight edge on pronoun disambiguation.

---

## Results

### Main Results

| Metric | Curriculum | Random | Winner |
|---|---|---|---|
| **Test Perplexity** ↓ | **28.7251** | 28.8082 | ✅ Curriculum |
| **Bits-per-Character (BPC)** ↓ | **4.8442** | 4.8484 | ✅ Curriculum |
| **LAMBADA Accuracy** ↑ | **0.08%** (4/5000) | 0.06% (3/5000) | ✅ Curriculum |
| **HellaSwag Accuracy** ↑ | **27.50%** (2750/10000) | 27.43% (2743/10000) | ✅ Curriculum |
| **OpenBookQA Accuracy** ↑ | 24.40% (122/500) | **26.60%** (133/500) | ✅ Random |
| **ARC-Easy Accuracy** ↑ | 34.04% (399/1172) | **34.98%** (410/1172) | ✅ Random |
| **WinoGrande Accuracy** ↑ | **51.93%** (658/1267) | 51.07% (647/1267) | ✅ Curriculum |

**Summary:** Curriculum wins PPL (1/1), Accuracy (3/5), ECE (3/5).

### Calibration — Expected Calibration Error (lower = better calibrated)

| Benchmark | Curriculum ECE | Random ECE | Winner |
|---|---|---|---|
| LAMBADA | **24.54%** | 27.70% | ✅ Curriculum |
| HellaSwag | **11.97%** | 12.04% | ✅ Curriculum |
| OpenBookQA | 37.75% | **34.87%** | ✅ Random |
| ARC-Easy | 22.49% | **21.26%** | ✅ Random |
| WinoGrande | **9.31%** | 10.17% | ✅ Curriculum |

---

## Statistical Analysis

Statistical significance was tested using **McNemar's test** on per-example correct/incorrect outcomes, and uncertainty was quantified with **bootstrap 95% confidence intervals** (2,000 resamples).

### Perplexity Confidence Intervals

| Model | 95% CI |
|---|---|
| Curriculum | [28.3461, 29.1077] |
| Random | [28.4230, 29.1868] |

The CIs overlap substantially, indicating the perplexity difference — while real — is small in absolute terms.

### Benchmark Significance Tests

| Benchmark | Curriculum CI | Random CI | McNemar p | Significant? |
|---|---|---|---|---|
| LAMBADA | [0.0%, 0.2%] | [0.0%, 0.1%] | 1.0000 | ❌ No |
| HellaSwag | [26.7%, 28.4%] | [26.6%, 28.3%] | 0.8187 | ❌ No |
| OpenBookQA | [20.6%, 28.4%] | [22.6%, 30.6%] | 0.1447 | ❌ No |
| ARC-Easy | [31.4%, 36.8%] | [32.3%, 37.6%] | 0.3859 | ❌ No |
| WinoGrande | [49.1%, 54.8%] | [48.3%, 53.7%] | 0.4932 | ❌ No |

**Conclusion:** No benchmark difference is statistically significant at α = 0.05. The curriculum model shows a consistent directional advantage on 3 of 5 benchmarks and perplexity, but the effect sizes are small relative to the noise at this model and data scale. Longer training (more epochs with a proper stage schedule) or a larger model may be needed to see curriculum learning benefits emerge more clearly.

---

## Reproducibility

All experiments are fully reproducible:

- Global random seed: **42** (applied to Python `random`, NumPy, and PyTorch)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- Device auto-detection: **CUDA → Apple MPS → CPU** (results may vary across hardware due to floating-point ordering)

To reproduce exactly, use the same GPU model and CUDA version as the original run.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
tokenizers>=0.15.0
datasets>=2.14.0
textstat>=0.7.3
tqdm>=4.65.0
psutil>=5.9.0
scipy
transformers
nltk
```

Install via Docker (recommended):
```bash
docker build -t nlp-curriculum .
```

Or manually:
```bash
pip install -r requirements.txt
```