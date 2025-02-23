from datasets import load_dataset
import os
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
fineweb_edu = "fineweb_edu"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# encode with tiktoken gpt4 bpe
enc = tiktoken.get_encoding("cl100k_base")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of int32 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.int32)

def save_shard(filename, tokens):
    np.save(filename, tokens)
    
nprocs = max(1, os.cpu_count()//2)

with mp.Pool(nprocs) as pool:
    shard_idx = 0
    # pre-allocate the buffer for shard
    shard_buffer = np.empty(shard_size, dtype=np.int32)
    token_cnt = 0
    progress_bar = None
    # each process will handle a chunk size of 16 documents to tokenize them in parallel
    # tokens in the for loop are the tokenized results for 16 * nprocs documents
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_cnt + len(tokens) < shard_size:
            shard_buffer[token_cnt:token_cnt+len(tokens)] = tokens
            token_cnt += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard: {shard_idx:06d}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_idx == 0 else "train"
            reminder = shard_size - token_cnt
            progress_bar.update(reminder)
            shard_buffer[token_cnt:token_cnt+reminder] = tokens[:reminder]
            file_path = os.path.join(DATA_CACHE_DIR, f"{fineweb_edu}_{split}_{shard_idx:06d}")
            # save the full shard
            save_shard(file_path, shard_buffer)
            shard_idx += 1
            token_cnt = len(tokens) - reminder
            shard_buffer[:token_cnt] = tokens[reminder:]
            progress_bar = None
    if token_cnt > 0:
        split = "val" if shard_idx == 0 else "train"
        file_path = os.path.join(DATA_CACHE_DIR, f"{fineweb_edu}_{split}_{shard_idx:06d}")
        save_shard(file_path, shard_buffer[:token_cnt])


