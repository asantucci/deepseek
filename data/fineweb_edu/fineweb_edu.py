import argparse
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from datasets import load_dataset
import tiktoken

parser = argparse.ArgumentParser(
    prog="Downloads & Tokenizes FineWeb 10BT Sample",
    usage="python3 fineweb_edu.py --cache_dir='<dir>'"
)
parser.add_argument('--cache_dir', type=str, default='~/data/edu_fineweb10B')
parser.add_argument('--shard_prefix', type=str, default='fineweb_edu')
parser.add_argument('--tokens_per_shard', type=int, default=int(1e8))  # With 100M tokens/shard, there are ~100 shards using our FineWeb10BT sample.
args = parser.parse_args()
args.cache_dir = os.path.expanduser(args.cache_dir)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

remote_name = "sample-10BT"
os.makedirs(args.cache_dir, exist_ok=True)

# Encode with tiktoken gpt4 bpe.
enc = tiktoken.get_encoding("cl100k_base")
eot = enc._special_tokens['<|endoftext|>'] # End of text token.

def tokenize(doc):
    """Tokenize a single document with <|endoftext|> prepended."""
    tokens = [eot] # The special <|endoftext|> token delimits all documents.
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.int32)

def save_shard(path, tokens):
    """Saves token array to disk."""
    np.save(path, tokens)

def process_and_shard(dataset, shard_size, out_dir, prefix):
    """Tokenizes a dataset and shards."""
    nprocs = max(1, os.cpu_count() // 2)
    shard_idx = 0
    token_count = 0
    shard_buffer = np.empty(shard_size, dtype=np.int32)
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            remaining = shard_size - token_count

            if len(tokens) <= remaining:
                shard_buffer[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard: {shard_idx:06d}")
                progress_bar.update(len(tokens))
            else:
                # Fills the remaining part of the current shard.
                shard_buffer[token_count:] = tokens[:remaining]
                progress_bar.update(remaining)

                # Save the completed shard.
                split = "val" if shard_idx == 0 else "train"
                shard_path = os.path.join(out_dir, f"{prefix}_{split}_{shard_idx:06d}")
                save_shard(shard_path, shard_buffer)

                shard_idx += 1
                progress_bar.close()
                progress_bar = None

                # Starts new shard with leftover tokens.
                leftover = tokens[remaining:]
                token_count = len(leftover)
                shard_buffer[:token_count] = leftover

        # Saves any remaining tokens in the last shard.
        if token_count > 0:
            print(f'Now saving the remaining tokens in final shard.')
            split = "val" if shard_idx == 0 else "train"
            shard_path = os.path.join(out_dir, f"{prefix}_{split}_{shard_idx:06d}")
            save_shard(shard_path, shard_buffer[:token_count])
            if progress_bar:
                progress_bar.close()


if __name__ == '__main__':
    process_and_shard(dataset, args.tokens_per_shard, args.cache_dir, args.shard_prefix)