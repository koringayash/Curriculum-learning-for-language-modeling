# """Main training configuration. Architecture, hyperparameters, curriculum settings."""
# import os
# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# import config_global

# # 📐 GPT Architecture (Exact from original)
# SEQ_LEN = 512
# N_EMBD = 768
# N_HEAD = 8
# N_LAYER = 12
# DROPOUT = 0.1
# VOCAB_SIZE = 30000

# # SEQ_LEN = 512
# # N_EMBD = 512
# # N_HEAD = 8
# # N_LAYER = 8
# # DROPOUT = 0.1
# # VOCAB_SIZE = 30000

# # 🏋️ Training
# BATCH_SIZE = 8
# ACCUMULATION_STEPS = 4
# LEARNING_RATE = 2.5e-4
# WEIGHT_DECAY = 0.01
# EPOCHS = 3  # 5
# GRAD_CLIP = 1.0

# # 📊 Checkpointing
# # Curriculum mode: saves at 25%, 50%, 75%, 100% of total steps (4 milestone snapshots kept permanently)
# #                  curriculum_25pct.pt | curriculum_50pct.pt | curriculum_75pct.pt | curriculum_final.pt
# # Random mode    : saves after each epoch, deletes previous → 1 final file (random_final.pt)
# # Total after full experiment: 5 model files

# # 📚 Curriculum
# USE_CURRICULUM = True
# CURRICULUM_BLEND_FRACTION = 0.05
# CURRICULUM_STAGE_WEIGHTS = {
#     1: [0.90, 0.10, 0.00, 0.00],
#     2: [0.15, 0.75, 0.10, 0.00],
#     3: [0.05, 0.15, 0.70, 0.10],
#     4: [0.05, 0.05, 0.15, 0.75]
# }

# # 📁 Paths
# DATA_DIR = os.path.join(config_global.PROJECT_ROOT, "data")
# CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")
# STAGE_BINS = {
#     1: os.path.join(DATA_DIR, "stage_1_easy.bin"),
#     2: os.path.join(DATA_DIR, "stage_2_medium.bin"),
#     3: os.path.join(DATA_DIR, "stage_3_hard.bin"),
#     4: os.path.join(DATA_DIR, "stage_4_very_hard.bin"),
# }

"""Main training configuration. Architecture, hyperparameters, curriculum settings."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global

# 📐 GPT Architecture
SEQ_LEN    = 512
N_EMBD     = 768
N_HEAD     = 8
N_LAYER    = 12
DROPOUT    = 0.1
VOCAB_SIZE = 30000

# 🏋️ Training
BATCH_SIZE         = 8
ACCUMULATION_STEPS = 4
LEARNING_RATE      = 2.5e-4
WEIGHT_DECAY       = 0.01
EPOCHS             = 1
GRAD_CLIP          = 1.0

# 📚 Curriculum — which stages are active each epoch
# Epoch 1: easy only          → model sees simple language first
# Epoch 2: easy + medium      → gradually introduced to harder text
# Epoch 3: all four stages    → full difficulty range
#
# Each epoch iterates over ALL documents in the included stages (no sampling).
# For random mode this setting is ignored — all 4 stages are used every epoch.
CURRICULUM_EPOCH_STAGES = {
    # 1: [1],
    # 2: [1, 2],
    # 3: [1, 2, 3],
    1: [1, 2, 3, 4],
}

# 📁 Paths
DATA_DIR       = os.path.join(config_global.PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")
STAGE_BINS = {
    1: os.path.join(DATA_DIR, "stage_1_easy.bin"),
    2: os.path.join(DATA_DIR, "stage_2_medium.bin"),
    3: os.path.join(DATA_DIR, "stage_3_hard.bin"),
    4: os.path.join(DATA_DIR, "stage_4_very_hard.bin"),
}