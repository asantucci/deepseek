import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from config import DeepSeekConfig


class Distributor(object):
    def __init__(self, gates: torch.tensor, topk: int):
        super().__init__()
        # gates is [B*T, num_experts]
        self.topk = topk
        # [B*T*topk, 2]
        batch_and_experts_indices = torch.nonzero(gates)
        # sort the batch and experts indices along the first dimension, batch_and_experts_indices is a list
        # of tuples, where the first element is the batch index and the second element is the expert index
        # by sorting along the first dimension, we will let the same assigned expert index be adjacent
        # and then we will use this order to reorder the input tensors
        # then finally after splitting the recordered input tensors, the first group is for first expert,
        # the second group is for second expert, etc.
        # use stable sort to guarantee the relative order of the same element so that we could get the weight for the expert output
        # with colume-wise order from left to right
        # [B*topk, 2]
        sorted_experts, index_sorted_experts = batch_and_experts_indices.sort(
            dim=0, stable=True
        )
        # get the order indices before sorting
        # [B*T*topk] one dimension tensor
        old_expert_indices = index_sorted_experts[:, 1]
        # find the batch index from the order of sorted experts
        # it will be used for the input tensors to make sure the tokens that assigned to the same expert are adjacent
        # and then use the _groups to split the input tensors
        # [B*T*topk] one dimension tensor
        self._batch_indices = batch_and_experts_indices[:, 0][old_expert_indices]
        # get the number of tokens assigned for each expert
        # [num_experts] one dimension tensor
        self._groups = (gates > 0).sum(dim=0).tolist()
        # get the weights for each expert output. It just get the non zero elements from the gates for each column from left to right
        # [B*T*topk, 1]
        self._weights = gates.t().reshape(-1)[gates.t().reshape(-1) > 0].view(-1, 1)

    def prepare_inputs_for_experts(self, x: torch.tensor) -> list[torch.tensor]:
        expanded_x = x[self._batch_indices]
        return expanded_x.split(self._groups)

    def combine(self, expert_outputs: list[torch.tensor]) -> torch.tensor:
        # [B*topk, d_model]
        combined_output = torch.cat(expert_outputs, dim=0)
        # apply the weights to the expert outputs
        # [B*topk, d_model]
        combined_output = combined_output * self._weights
        # use index_add to add results for each token and the index is _batch_indices
        # [B, d_model]
        output = torch.zeros(
            combined_output.shape[0] // self.topk,
            combined_output.shape[1],
            dtype=combined_output.dtype,
        ).to(combined_output.device)
        output.index_add_(0, self._batch_indices, combined_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dimension: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dimension, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dimension, bias=False)
        self.down_proj = nn.Linear(hidden_dimension, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor):
        # this is different from the FeedForward in Transformer paper
        # not sure why DeepSeek use the gate_proj instead of activation(up_proj(x))
        # to be able to load the checkpoint, follow their implementation at
        # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L389
        x = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        return x


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    Copied from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py#L500
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            # when we requuired aux loss, this grad loss is for the gradient of the second input of forward
            # which is the auxiliary loss
            # effectively since the grad is 1, the aux loss is added to the loss
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()

        self.num_shared_experts = config.num_shared_experts
        self.moe_hidden_dimension = config.moe_hidden_dimension

        self.topk = config.topk
        self.num_routed_experts = config.num_routed_experts

        # the weights intialization for deepseek is 0.006
        # from https://arxiv.org/abs/2401.06066
        # the number of potential routed experts is total_num_experts * num_smaller_experts_per_expert - num_shared_experts
        self.experts_weights = nn.Parameter(
            torch.randn(
                self.num_routed_experts,
                config.d_model,
            )
            * 0.006
        )

        # routed experts
        self.experts = nn.ModuleList(
            [
                FeedForward(config.d_model, self.moe_hidden_dimension)
                for _ in range(self.num_routed_experts)
            ]
        )

        self.shared_experts = FeedForward(
            config.d_model, self.moe_hidden_dimension * self.num_shared_experts
        )

        self.topk_norm_epsilon = config.topk_norm_epsilon
        self.normalized_moe_gates = config.normalized_moe_gates
        self.expert_load_balance_factor = config.expert_load_balance_factor

    def forward(self, x: torch.tensor, use_optimization: bool = True) -> torch.tensor:
        """
        x: tensor of shape [B, T, d_model]
        """
        return (
            self._forward_optimized(x) if use_optimization else self._forward_forloop(x)
        )

    def _forward_forloop(self, x: torch.tensor):
        """
        x: tensor of shape [B, T, d_model]
        """
        B, T = x.shape[0], x.shape[1]
        # [B* T, d_model]
        x = x.view(B * T, -1)
        # first get the output for the routed MoE and then added up the results from shared MoE
        # [B * T, total_routed_experts]
        routed_experts_output = F.linear(x, self.experts_weights)
        scores = F.softmax(routed_experts_output, dim=-1)
        # apply gate along the expert dimension to get the top k experts for each token
        # top_values: B * T, topk, this is the score for each expert
        # top_indices: B * T, topk, this is the index to find the corresponding expert
        top_values, top_indices = torch.topk(scores, k=self.topk, dim=-1, sorted=False)
        routed_experts_output = torch.zeros_like(x, dtype=x.dtype).to(x.device)
        for i in range(x.shape[0]):
            for j in range(self.topk):
                routed_experts_output[i] += (
                    self.experts[top_indices[i, j]](x[i]) * top_values[i, j]
                )

        shared_experts_output = self.shared_experts(x)

        # the output is sum of shared expert output and routed expert output plus the residual connection
        output = routed_experts_output + shared_experts_output
        output = output.view(B, T, -1)  # [B, T, d_model]

        return output

    def _forward_optimized(self, x: torch.tensor):
        """
        In the for loop implementation, the expert will transform the input tensor one by one.
        One optimization is to batch all input tensors for a given expert together and let the expert transform them with matrix multiplication.

        So in this implementation, we will
        1. first batch input tensors for each expert
        2. loop through each expert and transform its input tensors with matrix multiplication
        3. for each token, since it might be routed to multiple experts, we need to sum up its resutls with index_add function

        The reference for this implementation is
        https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py

        Args:
            x: tensor of shape [B, T, d_model]
        Returns:
            output: tensor of shape [B, T, d_model]
        """

        # combine the batch and time dimension
        B, T = x.shape[0], x.shape[1]
        # [B* T, d_model]
        x = x.view(B * T, -1)
        gates = F.linear(x, self.experts_weights)
        gates = F.softmax(gates, dim=-1)
        top_values, top_indices = torch.topk(gates, k=self.topk, dim=-1, sorted=False)
        # [B * T, num_experts]
        masked_gates = torch.zeros_like(gates, dtype=gates.dtype).to(gates.device)
        masked_gates = torch.scatter(masked_gates, 1, top_indices, top_values)
        if self.normalized_moe_gates:
            # renormalize the masked gates
            masked_gates = masked_gates / (
                masked_gates.sum(dim=-1, keepdim=True) + self.topk_norm_epsilon
            )
        distributor = Distributor(masked_gates, self.topk)
        routed_expert_inputs = distributor.prepare_inputs_for_experts(x)
        routed_expert_outputs = [
            self.experts[i](routed_expert_inputs[i])
            for i in range(self.num_routed_experts)
        ]
        # [B*T, d_model]
        routed_combined_outputs = distributor.combine(routed_expert_outputs)
        routed_combined_outputs = routed_combined_outputs.view(B, T, -1)
        if self.training:
            # get the expert load balance loss. The definition can be found in https://arxiv.org/abs/2401.06066
            masked_gates = masked_gates.view(B, T, -1)
            gates = gates.view(B, T, -1)
            load = (masked_gates > 0).sum(dim=1)
            expert_prob_sum = gates.sum(dim=1)
            expert_load_balance_loss = self.expert_load_balance_factor * (
                (self.num_routed_experts / (self.topk * T) * load)
                * (1.0 / T * expert_prob_sum)
            ).sum(dim=1)
            expert_load_balance_loss = expert_load_balance_loss.mean()
            routed_combined_outputs = AddAuxiliaryLoss.apply(
                routed_combined_outputs, expert_load_balance_loss
            )
        shared_expert_outputs = self.shared_experts(x).view(B, T, -1)
        output = routed_combined_outputs + shared_expert_outputs
        return output


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
        num_shared_experts=1,
        num_routed_experts=1,
        moe_hidden_dimension=20,
        mlp_hidden_dimension=20,
        topk=1,
        topk_norm_epsilon=1e-9,
        rms_norm_eps=1e-6,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        num_layers=1,
        vocab_size=10000,
        init_weight_std=0.006,
        first_k_dense_replace=0,
    )
    input = torch.randn(2, 2, 1024).to(config.device)
    model = MoE(config)
    model = model.to(config.device)
    output = model(input)
    print(f"MoE output shape: {output.shape}")
    sd = model.state_dict()
    sd_keys = sd.keys()
    for key in sd_keys:
        print(f"{key}: {sd[key].shape}")
