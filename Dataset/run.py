"""Dataset Module Runner. Executes Steps 1-5 sequentially with timing."""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config_global
from src.downloader import download_and_split
from src.tokenizer import train_tokenizer
from src.encoder import encode_corpus
from src.reference_trainer import train_reference_model
from src.scorer_stager import run as run_stager


sys.stdout = open('logs.txt', 'a')

def main():
    print("🚀 Starting Dataset Pipeline (Steps 1-5)...")
    config_global.set_seed()

    t0 = time.time()
    def timed_step(name, fn):
        print(f"\n⏳ Starting: {name}")
        t = time.time()
        fn()
        print(f"✅ Finished: {name} in {time.time() - t:.2f}s")

    timed_step("Download & Split", download_and_split)
    timed_step("Tokenizer Training", train_tokenizer)
    timed_step("Encoding Corpus", encode_corpus)
    timed_step("Reference Model Training", train_reference_model)
    timed_step("Scoring & Staging", run_stager)

    print(f"\n🎉 Dataset Pipeline Finished. Total Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()