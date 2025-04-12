# to format: black --line-length 88 sft.py

from dataclasses import dataclass
from tokenizer import Tokenizer
import datasets
from torch.utils.data import DataLoader, Dataset
import json
from training_config import TrainingConfig
from train import Trainer
from datacollator import DataCollatorForChatMl, ChatMlSpecialTokens

# Define a custom PyTorch Dataset
class SFTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["messages"]


if __name__ == "__main__":
    sft_training_config_file = "sft.json"
    with open(sft_training_config_file, "r") as f:
        training_config = json.load(f)
    sft_training_config = TrainingConfig(**training_config)
    
    tokenizer = Tokenizer(sft_training_config.tokenizer_type)
    tokenizer.add_special_tokens(
        [ChatMlSpecialTokens().bos_token, ChatMlSpecialTokens().eos_token]
    )
    # load huggingface dataset
    train_dataset = datasets.load_dataset(sft_training_config.dataset_name, split=sft_training_config.train_split)
    eval_dataset = datasets.load_dataset(sft_training_config.dataset_name, split=sft_training_config.eval_split)

    sft_train_dataset = SFTDataset(train_dataset)
    sft_eval_dataset = SFTDataset(eval_dataset)

    # create the datacollator
    data_collator = DataCollatorForChatMl(
        tokenizer,
        tokenizer.eos_token_id,
        # pytorch cross entropy loss will ignore labels with value -100
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        -100,
        ChatMlSpecialTokens().assistant,
        tokenizer.eos_token_id,
    )

    # create dataloader
    sft_train_dataloader = DataLoader(
        sft_train_dataset,
        batch_size=sft_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )
    sft_eval_dataloader = DataLoader(
        sft_eval_dataset,
        batch_size=sft_training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )
    
    # max_len = 0
    # train_batch_size = 0
    # eval_batch_size = 0
    # for batch in sft_train_dataloader:
    #     input_ids = batch['input_ids']
    #     max_len = max(max_len, input_ids.shape[1])
    #     train_batch_size += 1
    #     if train_batch_size % 100 == 0:
    #         print(f"train_batch_size: {train_batch_size}")
    # for batch in sft_eval_dataloader:
    #     input_ids = batch['input_ids']
    #     max_len = max(max_len, input_ids.shape[1])
    #     eval_batch_size += 1
    #     if eval_batch_size % 100 == 0:
    #         print(f"eval_batch_size: {eval_batch_size}")
    # print(f"max_len: {max_len}")
    
    sft_trainer = Trainer(
        sft_train_dataloader,
        sft_eval_dataloader,
        sft_training_config,
    )
    sft_trainer.train()
