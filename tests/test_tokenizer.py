import pytest

from bpe_tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer(tokenizer_type="cl100k_base")

def test_basic_encoding_decoding(tokenizer):
    text = "Hello, how are you?"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)
    assert decoded == text

def test_get_token_id_and_string(tokenizer):
    token_ids = tokenizer.get_token_id("Hello")
    assert isinstance(token_ids, list)
    assert len(token_ids)
    token_str = tokenizer.get_token_str(token_ids[0])
    assert isinstance(token_str, str)

def test_add_special_tokens(tokenizer):
    vocab_before = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    vocab_after = tokenizer.get_vocab_size()

    # Ensure vocab size increased
    assert vocab_after >= vocab_before + 2

    # Encode a string with a special token
    text = "test <|im_start|>"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert "<|im_start|>" in decoded
