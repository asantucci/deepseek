from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device: str = "cuda"
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    eval_iters: int = 5
    eval_interval: int = 10
    dtype: str = "bfloat16"
    measure_throughput_interval: int = 100
    estimate_throughput: bool = False
    wandb_log: bool = True
    wandb_project: str = "deepseek training"
    wandb_run_name: str = "8_bit_optimizer"
    adamw_use_fused: bool = True
    max_train_steps: int = 30000
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_iters: int = 500
    lr_decay_iters: int = 1000
    decay_lr: bool = True
    out_dir: str = "output"
    resume: bool = False
    checkpoint_path: str = "8_bit_optimizer_ckpt.pt"
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    use_eight_bit_optimizer: bool = True
    grad_clip: float = 1.0
    model_config_path: str = "config.json"
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str = "train_sft"
    eval_split: str = "test_sft"
    tokenizer_type: str = "cl100k_base"
