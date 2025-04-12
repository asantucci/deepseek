import jinja2

from dataclasses import dataclass
import torch
from tokenizer import Tokenizer

# Reference: https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L44.
@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        """
        the jinja2 template for the chatml format
        """
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


def format_input_text(input: list[dict[str, str]], add_generation_prompt: bool = False):
    """
    format the input text with chat template to differentiate among different roles

    an exmple of input:
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    when add_generation_prompt is False, the output should be:
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm fine, thank you!<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>

    when add_generation_prompt is True, the output should be:
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm fine, thank you!<|im_end|>
    <|im_start|>assistant
    """

    template = jinja2.Template(ChatMlSpecialTokens().chat_template)
    return template.render(messages=input, add_generation_prompt=add_generation_prompt)

def pad(examples: list[torch.Tensor], pad_value: int):
    """
    pad the input text to the max length of the batch
    """
    max_length = max(len(example) for example in examples)
    padded_examples = [
        torch.cat([example, torch.full((max_length - len(example),), pad_value)])
        for example in examples
    ]
    return padded_examples

class DataCollatorForChatMl:
    """
    The data collator will primary do three things:
    1. format the input text with chat template
    2. pad the input text to the max length of the batch
    3. set the label values for the non-assistant response tokens as ignore_index
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        pad_token_id: int,
        ignore_index: int,
        assistant_response_format: str,
        end_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.assistant_response_format = assistant_response_format
        self.end_token_id = end_token_id

    def process(self, examples: list[list[dict[str, str]]]):
        """
        process a batch of examples
        """
        formatted_examples = [format_input_text(example) for example in examples]
        tokenized_examples = [
            self.tokenizer.encode(example) for example in formatted_examples
        ]
        input_ids = [torch.tensor(example[:-1]) for example in tokenized_examples]
        attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
        labels = [torch.tensor(example[1:]) for example in tokenized_examples]

        input_ids = pad(input_ids, self.pad_token_id)
        attention_mask = pad(attention_mask, 0)
        labels = pad(labels, self.ignore_index)
        # mask out the non-assistant response tokens in labels
        labels = self.mask_labels(labels)
        batch = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }
        return batch

    def mask_labels(self, labels: list[torch.Tensor]):
        """
        mask the labels for the non-assistant response tokens
        """
        response_ids = self.tokenizer.encode(self.assistant_response_format)
        for label in labels:
            start_ind = 0
            prev_assistant_response = False
            i = 0
            while i < len(label):
                if i < len(label) - len(response_ids) and torch.equal(
                    label[i : i + len(response_ids)], torch.tensor(response_ids)
                ):
                    label[start_ind : i + len(response_ids)] = self.ignore_index
                    i += len(response_ids)
                    prev_assistant_response = True
                elif (
                    torch.equal(label[i], torch.tensor(self.end_token_id))
                    and prev_assistant_response
                ):
                    start_ind = i + 1
                    prev_assistant_response = False
                    i += 1
                else:
                    i += 1
            label[start_ind:] = self.ignore_index
        return labels
    
if __name__ == "__main__":
    tokenizer = Tokenizer('cl100k_base')
    tokenizer.add_special_tokens(
        [ChatMlSpecialTokens().bos_token, ChatMlSpecialTokens().eos_token]
    )
    messages = [
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

    formatted_input = format_input_text(messages[0])
    print(f"formatted_input: {formatted_input}")

    token_ids = tokenizer.encode(formatted_input)
    print(f"token_ids: {token_ids}")

    token_strs = tokenizer.decode(token_ids)
    print(f"token_strs: {token_strs}")

    bos = ChatMlSpecialTokens().bos_token
    bos_encoded = tokenizer.encode(bos)
    print(f"bos_encoded: {bos_encoded}")

    eos = ChatMlSpecialTokens().eos_token
    eos_encoded = tokenizer.encode(eos)[0]
    print(f"eos_encoded: {eos_encoded}")

    data_collator = DataCollatorForChatMl(
        tokenizer,
        tokenizer.eos_token_id,
        -100,
        ChatMlSpecialTokens().assistant,
        tokenizer.eos_token_id,
    )
    batch = data_collator.process(messages)
    print(f"batch: {batch}")