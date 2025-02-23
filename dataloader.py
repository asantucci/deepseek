import os
import random
import numpy as np
import torch

def load_shard(file_path):
    '''
    load a shard from the given path and convert to pytorch tensor
    '''
    # need to use int64 so that torch cross entropy loss can work
    np_tensors = np.load(file_path).astype(np.int64)
    return torch.from_numpy(np_tensors)

class DataLoader:
    def __init__(self, data_dir, batch_size, seq_len, split, device="cpu", shuffle=True):
        '''
        data_dir: the directory of the data
        batch_size: the batch size
        seq_len: the sequence length
        split: the split of the data, train or val
        shuffle: whether to shuffle the shards
        '''
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.shuffle = shuffle
        self.device = device
        all_shards = os.listdir(data_dir)
        self.shards = [shard for shard in all_shards if split in shard]
        if shuffle:
            random.shuffle(self.shards)
        self.current_shard = 0
        self.current_pos = 0
        self.reset_status()
        
        
    def reset_status(self):
        '''
        reset the status of the dataloader
        '''
        self.current_shard = 0
        self.current_pos = 0
        if self.shuffle:
            random.shuffle(self.shards)
        self.tokens = load_shard(os.path.join(self.data_dir, self.shards[self.current_shard]))
    
    def next_batch(self):
        '''
        return the next batch of data
        '''
        batch_tokens = self.tokens[self.current_pos:self.current_pos+self.batch_size * self.seq_len+1]
        x = batch_tokens[:-1].view(self.batch_size, self.seq_len)
        y = batch_tokens[1:].view(self.batch_size, self.seq_len)
        self.current_pos += self.batch_size * self.seq_len
        if self.current_pos + self.batch_size * self.seq_len + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_pos = 0
            self.tokens = load_shard(os.path.join(self.data_dir, self.shards[self.current_shard]))
        return x.to(self.device), y.to(self.device)
    

class FineWebEduDataLoader(DataLoader):
    def __init__(self, batch_size, seq_len, split, device="cpu", shuffle=True):
        data_dir = os.path.join(os.path.dirname(__file__), "data", "fineweb_edu", "edu_fineweb10B")
        print(f"Loading data from {data_dir}")
        super().__init__(data_dir, batch_size, seq_len, split, device, shuffle)

class TinyShakespeareDataLoader(DataLoader):
    def __init__(self, batch_size, seq_len, split, device="cpu", shuffle=True):
        data_dir = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare")
        print(f"Loading data from {data_dir}")
        super().__init__(data_dir, batch_size, seq_len, split, device, shuffle)
    
    def reset_status(self):
        pass
    
    def next_batch(self):
        if self.split == "train":
            data = np.memmap(os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.seq_len, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.seq_len]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.seq_len]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            # make the tensors in the non-pageable area
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    