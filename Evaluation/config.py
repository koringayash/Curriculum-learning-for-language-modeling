# import os, sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# import config_global
# DATA_DIR = os.path.join(config_global.PROJECT_ROOT, "data")
# CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")
# EVAL_MAX_LAMBADA = 500
# EVAL_MAX_HELLASWAG = 500


"""
Evaluation/config.py
Module-specific configuration for model evaluation and benchmarking.
Contains only evaluation hyperparameters. Paths are inherited from 
Model/config.py and config_global.py to avoid duplication.
"""
import os
import sys

# Add project root to sys.path for global config access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global

# 🎯 Benchmark Sampling Limits
# Production defaults: LAMBADA ~5,000 | HellaSwag ~10,000
# Lower values provided for faster iteration/debugging
EVAL_MAX_LAMBADA = 5000
EVAL_MAX_HELLASWAG = 10000

# ⚙️ Evaluation Settings
EVAL_SEQ_LEN = 512          # Must match or exceed the model's training SEQ_LEN
EVAL_PIN_MEMORY = True if config_global.DEVICE == "cuda" else False

CHECKPOINT_DIR = os.path.join(config_global.PROJECT_ROOT, "checkpoints")