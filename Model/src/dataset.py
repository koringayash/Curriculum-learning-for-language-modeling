"""dataset.py: Memory-mapped PyTorch Dataset for tokenized .bin files."""
import numpy as np
import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, bin_file: str, seq_len: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = (len(self.data) - 1) // seq_len
    def __len__(self) -> int: return self.n
    def __getitem__(self, idx: int) -> tuple:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return torch.from_numpy(chunk[:-1].astype(np.int64)), torch.from_numpy(chunk[1:].astype(np.int64))