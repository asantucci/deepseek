import os
import tempfile
import numpy as np
import torch
import pytest

from dataloader import DataLoader, FineWebEduDataLoader, TinyShakespeareDataLoader, load_shard

@pytest.fixture
def shard_data():
    """Create a temporary shard file with synthetic token data."""
    data = np.arange(1000, dtype=np.int64)
    tmpdir = tempfile.mkdtemp()
    shard_path = os.path.join(tmpdir, "train_000.npy")
    np.save(shard_path, data)
    return tmpdir

def test_load_shard_returns_correct_tensor_type(shard_data):
    path = os.path.join(shard_data, "train_000.npy")
    tensor = load_shard(path)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.int64

def test_dataloader_batch_shape_and_alignment(shard_data):
    loader = DataLoader(
        data_dir=shard_data,
        batch_size=4,
        seq_len=8,
        split="train",
        device="cpu",
        shuffle=False,
    )

    x, y = loader.next_batch()
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # Check alignment: y should be shifted by 1 from x
    assert torch.all(y[:, :-1] == x[:, 1:])

def test_dataloader_wraps_to_next_shard_without_crash(shard_data):
    # Save a second shard to test wraparound logic
    np.save(os.path.join(shard_data, "train_001.npy"), np.arange(1000, 2000, dtype=np.int64))
    loader = DataLoader(
        data_dir=shard_data,
        batch_size=16,
        seq_len=20,
        split="train",
        device="cpu",
        shuffle=False,
    )
    # Consume tokens until rollover
    for _ in range(20):
        x, y = loader.next_batch()
        assert x.shape == (16, 20)

def test_fineweb_loader_initializes(monkeypatch, shard_data):
    monkeypatch.setenv("DISABLE_PRINT", "1")  # optional if you want to suppress loader prints
    loader = FineWebEduDataLoader(
        prefix_dir=shard_data,
        batch_size=2,
        seq_len=4,
        split="train",
        device="cpu",
        shuffle=True,
    )
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 4)

def test_tinyshakespeare_batch_shapes(tmp_path):
    # Create dummy binary files
    data_dir = tmp_path / "data" / "tinyshakespeare"
    data_dir.mkdir(parents=True)
    tokens = np.arange(2048, dtype=np.uint16)
    with open(data_dir / "train.bin", "wb") as f:
        tokens.tofile(f)

    # Patch __file__ location to simulate `TinyShakespeareDataLoader` base path
    import dataloader as dl
    dl.__file__ = str(tmp_path / "fake_script_location.py")

    loader = TinyShakespeareDataLoader(
        batch_size=2,
        seq_len=64,
        split="train",
        device="cpu"
    )
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 64)
