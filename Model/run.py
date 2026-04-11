"""Model Module Runner. CLI entry for Step 6."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train GPT with Curriculum or Random sampling")
    parser.add_argument("--mode", choices=["curriculum", "random"], default="curriculum", help="Training mode")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(mode=args.mode, resume=args.resume)

if __name__ == "__main__":
    main()