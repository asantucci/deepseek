import os
import random
import numpy as np
import torch

def load_shard(file_path):
    """
    Load a single .npy shard from disk and return as torch tensor.
    CrossEntropyLoss expects torch.int64 dtype for targets.
    """
    array = np.load(file_path).astype(np.int64)
    return torch.from_numpy(array)

class DataLoader:
    def __init__(self, data_dir, batch_size, seq_len, split, device="cpu", shuffle=True):
        """
        Parameters:
            data_dir (str): Path to directory containing dataset shards.
            batch_size (int): Number of sequences per batch.
            seq_len (int): Length of each input sequence.
            split (str): 'train' or 'val'.
            device (str): 'cpu' or 'cuda'.
            shuffle (bool): Whether to shuffle shards on load/reset.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.device = device
        self.shuffle = shuffle

        all_shards = os.listdir(data_dir)
        self.shards = [f for f in all_shards if split in f]

        if shuffle:
            random.shuffle(self.shards)

        self.current_shard = 0
        self.current_pos = 0
        self.tokens = None
        self.reset_status()

    def reset_status(self):
        """Reset to beginning of a shuffled shard list."""
        self.current_shard = 0
        self.current_pos = 0
        if self.shuffle:
            random.shuffle(self.shards)
        shard_path = os.path.join(self.data_dir, self.shards[self.current_shard])
        self.tokens = load_shard(shard_path)

    def next_batch(self):
        """
        Return the next batch of input-output pairs (x, y), where:
            x = current tokens
            y = next tokens (shifted by 1)
        """
        span = self.batch_size * self.seq_len + 1
        batch_tokens = self.tokens[self.current_pos:self.current_pos + span]

        x = batch_tokens[:-1].view(self.batch_size, self.seq_len)
        y = batch_tokens[1:].view(self.batch_size, self.seq_len)

        self.current_pos += self.batch_size * self.seq_len

        if self.current_pos + span > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_pos = 0
            shard_path = os.path.join(self.data_dir, self.shards[self.current_shard])
            self.tokens = load_shard(shard_path)

        return x.to(self.device), y.to(self.device)


class FineWebEduDataLoader(DataLoader):
    def __init__(self, prefix_dir, batch_size, seq_len, split, device, shuffle=True):
        """
        Loads FineWeb EDU-formatted tokenized data from:
            {prefix_dir}/sample-10BT/
        """
        print(f"Loading FineWeb EDU data from {prefix_dir}")
        super().__init__(prefix_dir, batch_size, seq_len, split, device, shuffle)


class TinyShakespeareDataLoader(DataLoader):
    def __init__(self, batch_size, seq_len, split, device="cpu", shuffle=True):
        """
        Loads TinyShakespeare binary dataset from local `data/tinyshakespeare` directory.
        """
        data_dir = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare")
        print(f"Loading TinyShakespeare data from {data_dir}")
        super().__init__(data_dir, batch_size, seq_len, split, device, shuffle)

    def reset_status(self):
        """Override: TinyShakespeare uses mmap binary files, so no shard loading needed."""
        pass

    def next_batch(self):
        """Read batch from memory-mapped binary and return tokenized (x, y) tensors."""
        file_name = "train.bin" if self.split == "train" else "val.bin"
        mmap_path = os.path.join(self.data_dir, file_name)
        data = np.memmap(mmap_path, dtype=np.uint16, mode="r")

        # Random start indices for each sequence in the batch
        starts = torch.randint(0, len(data) - self.seq_len - 1, (self.batch_size,))

        x = torch.stack([
            torch.from_numpy(data[i:i + self.seq_len].astype(np.int64))
            for i in starts
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.seq_len].astype(np.int64))
            for i in starts
        ])

        if self.device == "cuda":
            # Enables async GPU transfer
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True)
            )
        return x.to(self.device), y.to(self.device)
