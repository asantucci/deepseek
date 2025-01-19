import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from config import DeepSeekConfig


class Distributor(object):
    def __init__(self, gates: torch.tensor, top_k: int):
        super().__init__()
        # gates is [B*T, num_experts]
        self.top_k = top_k
        # [B*T*top_k, 2]
        batch_and_experts_indices = torch.nonzero(gates)
        # sort the batch and experts indices along the first dimension, batch_and_experts_indices is a list
        # of tuples, where the first element is the batch index and the second element is the expert index
        # by sorting along the first dimension, we will let the same assigned expert index be adjacent
        # and then we will use this order to reorder the input tensors
        # then finally after splitting the recordered input tensors, the first group is for first expert,
        # the second group is for second expert, etc.
        # use stable sort to guarantee the relative order of the same element so that we could get the weight for the expert output
        # with colume-wise order from left to right
        # [B*top_k, 2]
        sorted_experts, index_sorted_experts = batch_and_experts_indices.sort(
            dim=0, stable=True
        )
        # get the order indices before sorting
        # [B*T*top_k] one dimension tensor
        old_expert_indices = index_sorted_experts[:, 1]
        # find the batch index from the order of sorted experts
        # it will be used for the input tensors to make sure the tokens that assigned to the same expert are adjacent
        # and then use the _groups to split the input tensors
        # [B*T*top_k] one dimension tensor
        self._batch_indices = batch_and_experts_indices[:, 0][old_expert_indices]
        # get the number of tokens assigned for each expert
        # [num_experts] one dimension tensor
        self._groups = (gates > 0).sum(dim=0).tolist()
        # get the weights for each expert output. It just get the non zero elements from the gates for each column from left to right
        # [B*T*top_k, 1]
        self._weights = gates.t().reshape(-1)[gates.t().reshape(-1) > 0].view(-1, 1)

    def prepare_inputs_for_experts(self, x: torch.tensor) -> list[torch.tensor]:
        expanded_x = x[self._batch_indices]
        return expanded_x.split(self._groups)

    def combine(self, expert_outputs: list[torch.tensor]) -> torch.tensor:
        # [B*top_k, d_model]
        combined_output = torch.cat(expert_outputs, dim=0)
        # apply the weights to the expert outputs
        # [B*top_k, d_model]
        combined_output = combined_output * self._weights
        # use index_add to add results for each token and the index is _batch_indices
        # [B, d_model]
        output = torch.zeros(
            combined_output.shape[0] // self.top_k,
            combined_output.shape[1],
            dtype=combined_output.dtype,
        ).to(combined_output.device)
        output.index_add_(0, self._batch_indices, combined_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dimension, bias=False)
        self.linear2 = nn.Linear(hidden_dimension, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.rms_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = self.linear1(self.rms_norm(x))
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()

        self.num_shared_experts = config.num_shared_experts
        hidden_dimension_per_expert = (
            config.hidden_dimension // config.num_smaller_experts_per_expert
        )

        self.top_k = (
            config.num_activated_experts * config.num_smaller_experts_per_expert
            - config.num_shared_experts
        )
        self.total_routed_experts = (
            config.total_num_experts * config.num_smaller_experts_per_expert
            - self.num_shared_experts
        )

        # the weights intialization for deepseek is 0.006
        # from https://arxiv.org/abs/2401.06066
        # the number of potential routed experts is total_num_experts * num_smaller_experts_per_expert - num_shared_experts
        self.experts_weights = nn.Parameter(
            torch.randn(
                config.d_model,
                self.total_routed_experts,
                requires_grad=True,
            )
            * 0.006
        )

        self.routed_experts = nn.ModuleList(
            [
                FeedForward(config.d_model, hidden_dimension_per_expert, config.dropout)
                for _ in range(self.total_routed_experts)
            ]
        )

        self.shared_experts = nn.ModuleList(
            [
                FeedForward(config.d_model, hidden_dimension_per_expert, config.dropout)
                for _ in range(self.num_shared_experts)
            ]
        )

        self.epsilon = config.epsilon
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
        routed_experts_output = x @ self.experts_weights
        scores = F.softmax(routed_experts_output, dim=-1)
        # apply gate along the expert dimension to get the top k experts for each token
        # top_values: B * T, top_k, this is the score for each expert
        # top_indices: B * T, top_k, this is the index to find the corresponding expert
        top_values, top_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        routed_experts_output = torch.zeros_like(x, dtype=x.dtype).to(x.device)
        for i in range(x.shape[0]):
            for j in range(self.top_k):
                routed_experts_output[i] += (
                    self.routed_experts[top_indices[i, j]](x[i]) * top_values[i, j]
                )

        shared_experts_output = torch.zeros_like(x, dtype=x.dtype).to(x.device)
        # add the shared experts output
        for i in range(x.shape[0]):
            for j in range(self.num_shared_experts):
                shared_experts_output[i] += self.shared_experts[j](x[i])

        # the output is sum of shared expert output and routed expert output plus the residual connection
        output = routed_experts_output + shared_experts_output + x
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
        gates = x @ self.experts_weights
        gates = F.softmax(gates, dim=-1)
        top_values, top_indices = torch.topk(gates, k=self.top_k, dim=-1, sorted=False)
        # [B * T, num_experts]
        masked_gates = torch.zeros_like(gates, dtype=gates.dtype).to(gates.device)
        masked_gates = torch.scatter(masked_gates, 1, top_indices, top_values)
        # renormalize the masked gates
        masked_gates = masked_gates / (
            masked_gates.sum(dim=-1, keepdim=True) + self.epsilon
        )
        distributor = Distributor(masked_gates, self.top_k)
        routed_expert_inputs = distributor.prepare_inputs_for_experts(x)
        routed_expert_outputs = [
            self.routed_experts[i](routed_expert_inputs[i])
            for i in range(self.total_routed_experts)
        ]
        # [B*T, d_model]
        routed_combined_outputs = distributor.combine(routed_expert_outputs)
        shared_expert_outputs = torch.stack(
            [self.shared_experts[i](x) for i in range(self.num_shared_experts)]
        ).sum(dim=0)
        output = routed_combined_outputs + shared_expert_outputs

        # get the expert load balance loss. The definition can be found in https://arxiv.org/abs/2401.06066
        expert_load_balance_loss = None
        if self.training:
            masked_gates = masked_gates.view(B, T, -1)
            gates = gates.view(B, T, -1)
            load = (masked_gates > 0).sum(dim=1)
            expert_prob_sum = gates.sum(dim=1)
            expert_load_balance_loss = (
                self.expert_load_balance_factor
                * (
                    (self.total_routed_experts / (self.top_k * T) * load)
                    * (1.0 / T * expert_prob_sum)
                ).sum(dim=1)
            )
            expert_load_balance_loss = expert_load_balance_loss.mean()
        return output.view(B, T, -1), expert_load_balance_loss


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
        num_blocks=1,
        vocab_size=10000,
    )
    input = torch.randn(2, 2, 1024).to(config.device)
    moe = MoE(config)
    moe = moe.to(config.device)
    output, expert_load_balance_loss = moe(input)
    print(f"MoE output shape: {output.shape}")
    print(f"Expert load balance loss: {expert_load_balance_loss}")
