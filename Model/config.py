"""Main training configuration. Architecture, hyperparameters, curriculum settings."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global

# 📐 GPT Architecture (Exact from original)
SEQ_LEN = 512
N_EMBD = 768
N_HEAD = 8
N_LAYER = 12
DROPOUT = 0.1
VOCAB_SIZE = 30000

# SEQ_LEN = 512
# N_EMBD = 512
# N_HEAD = 8
# N_LAYER = 8
# DROPOUT = 0.1
# VOCAB_SIZE = 30000

# 🏋️ Training
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3  # 5
GRAD_CLIP = 1.0

# 📊 Checkpointing
SAVE_INTERVAL_PCT = 0.20  # 20%, 40%, 60%, 80%, 100%

# 📚 Curriculum
USE_CURRICULUM = True
CURRICULUM_BLEND_FRACTION = 0.05
CURRICULUM_STAGE_WEIGHTS = {
    1: [0.90, 0.10, 0.00, 0.00],
    2: [0.15, 0.75, 0.10, 0.00],
    3: [0.05, 0.15, 0.70, 0.10],
    4: [0.05, 0.05, 0.15, 0.75]
}

# 📁 Paths
DATA_DIR = os.path.join(config_global.PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")
STAGE_BINS = {
    i: os.path.join(DATA_DIR, f"stage_{i}.bin") for i in range(1, 5)
}