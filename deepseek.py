import torch
import torch.nn as nn
from moe import MoE
from mla import MultiHeadLatentAttention
from config import DeepSeekConfig
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.attn = MultiHeadLatentAttention(config)
        self.moe = MoE(config)  
    
    def forward(self, x: torch.tensor):
        x = x + self.attn(x)
        moe_output, expert_load_balance_loss = self.moe(x)
        x = x + moe_output
        return x, expert_load_balance_loss
    

class DeepSeek(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.lut = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.init_weight_std = config.init_weight_std
        # initialize weights
        self.apply(self._init_weights)
        
        
    def forward(self, x: torch.tensor, targets: torch.tensor=None) -> torch.tensor:
        '''
        Args:
            x: (B, T)
            target: (B, T)
        return: (B, T, vocab_size)
        '''
        B, T = x.shape
        assert T <= self.block_size, f"Sequence length {T} cannot exceed block size {self.block_size}"
        
        x = self.lut(x)
        x = self.dropout(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # for inference, only the last token logits is used for prediction the next token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_weight_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_weight_std)


if __name__ == "__main__":
    config = DeepSeekConfig(
        d_model=1024,
        nheads=128,
        block_size=512,
        dropout=0.0,
        device="cuda",
        use_kv_cache=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=128,
        num_shared_experts=3,
        total_num_experts=5,
        hidden_dimension=20,
        num_smaller_experts_per_expert=2,
        num_activated_experts=4,
        epsilon=1e-9,
        expert_load_balance_factor=0.01,
        vocab_size=10000,
        num_layers=2,
        init_weight_std=0.02,
    )
    input = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    targets = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    model = DeepSeek(config).to(config.device)
    output, loss = model(input, targets)
    print(f"when targets is not None: output shape: {output.shape}, loss: {loss}")
    targets = None
    model.eval()
    output, loss = model(input, targets)
    print(f"when targets is None: output shape: {output.shape}, loss: {loss}")
    # for name, module in model.named_modules():
    #         print(f"{name}: {module}")
