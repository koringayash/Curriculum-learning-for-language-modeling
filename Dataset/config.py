"""Dataset module configuration. Contains paths, hyperparameters, and scoring weights."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global

# 📊 Dataset Scale
TRAIN_DOCS = 1_500_000
TEST_DOCS = 5000
MIN_DOC_CHARS = 300
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "CC-MAIN-2024-10"
RANDOM_SEED = 42

# 📁 Paths
DATA_DIR = os.path.join(config_global.PROJECT_ROOT, "data")
TOKENIZER_DIR = os.path.join(config_global.PROJECT_ROOT, "tokenizer")
CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")

TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, "bpe_tokenizer.json")
TRAIN_JSONL = os.path.join(DATA_DIR, "train_raw.jsonl")
TEST_JSONL  = os.path.join(DATA_DIR, "test_raw.jsonl")
TRAIN_TOKENS = os.path.join(DATA_DIR, "train_tokens.bin")
TEST_TOKENS  = os.path.join(DATA_DIR, "test_tokens.bin")   # ✅ FIX: added missing path
DOC_OFFSETS = os.path.join(DATA_DIR, "doc_offsets.json")
STAGE_BOUNDARIES = os.path.join(DATA_DIR, "stage_boundaries.json")

# Output stage binaries (used by Model module)
STAGE_BINS = {
    1: os.path.join(DATA_DIR, "stage_1_easy.bin"),
    2: os.path.join(DATA_DIR, "stage_2_medium.bin"),
    3: os.path.join(DATA_DIR, "stage_3_hard.bin"),
    4: os.path.join(DATA_DIR, "stage_4_very_hard.bin"),
}

# 🎓 Reference Model Hyperparameters
REF_SEQ_LEN    = 256
REF_N_LAYER    = 2
REF_N_EMBD     = 128
REF_N_HEAD     = 4
REF_LR         = 3e-4
REF_EPOCHS     = 1
REF_BATCH_SIZE = 32

# ⚖️ Scoring Weights (sum to 1.0)
SCORE_WEIGHT_PPL   = 0.50
SCORE_WEIGHT_FLESCH = 0.30
SCORE_WEIGHT_TTR   = 0.10
SCORE_WEIGHT_COMP  = 0.10

VOCAB_SIZE = 30000