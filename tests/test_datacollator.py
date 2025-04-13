import pytest
import torch
from bpe_tokenizer import Tokenizer
from datacollator import (
    ChatMlSpecialTokens,
    format_input_text,
    DataCollatorForChatMl,
)

@pytest.fixture(scope="module")
def tokenizer():
    tok = Tokenizer("cl100k_base")
    tok.add_special_tokens([
        ChatMlSpecialTokens().bos_token,
        ChatMlSpecialTokens().eos_token,
    ])
    return tok

@pytest.fixture
def messages():
    return [
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "user", "content": "what is up?"},
            {"role": "assistant", "content": "not much, just chilling"},
            {"role": "user", "content": "what is your name?"},
        ],
    ]

def test_formatting_and_tokenization_roundtrip(tokenizer, messages):
    formatted = format_input_text(messages[0])
    token_ids = tokenizer.encode(formatted)
    decoded = tokenizer.decode(token_ids)
    assert ChatMlSpecialTokens().bos_token in decoded
    assert ChatMlSpecialTokens().eos_token in decoded

def test_data_collator_batch_shapes(tokenizer, messages):
    collator = DataCollatorForChatMl(
        tokenizer,
        tokenizer.eos_token_id,
        ignore_index=-100,
        assistant_response_format=ChatMlSpecialTokens().assistant,
        end_token_id=tokenizer.eos_token_id,
    )

    batch = collator.process(messages)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]

def test_label_masking_applies_to_non_assistant_only(tokenizer, messages):
    collator = DataCollatorForChatMl(
        tokenizer,
        tokenizer.eos_token_id,
        ignore_index=-100,
        assistant_response_format=ChatMlSpecialTokens().assistant,
        end_token_id=tokenizer.eos_token_id,
    )
    batch = collator.process(messages)
    labels = batch["labels"]
    # At least some tokens should be ignored
    assert (labels == -100).sum() > 0
    # Assistant responses should have some non-ignored tokens
    assert (labels != -100).any()
