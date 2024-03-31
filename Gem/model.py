import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

import config as gem_config


class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(self,
                embedding: torch.Tensor,
                hidden_states: torch.Tensor,
                output_position: torch.Tensor,
                temperatures: torch.Tensor,
                top_ps: torch.Tensor,
                tops_ks: torch.Tensor,
                embedding_bias) -> torch.Tensor:

        # on choisi  le dernier element de chaque sequence
        hidden_states = hidden_states.index_select(1, output_position).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        # applique la temperature
        logits.div_(temperatures.unsqueeze(dim=1))

        # calcule la temperature avec  SOFTMAX
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # on applique top-p et top-k
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        tops_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        tops_ks_mask = tops_ks_mask.expand(probs_idx.shape[0], -1)
        top_ps_mask = top_ps_mask >= tops_ks.unsqueeze(dim=1)
        probs_sort = torch.where(tops_ks_mask, 0, probs_sort)

        # renormalisation

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)

        return next_token_ids


def precompute_freqs_ids(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta**(torch.arange(0,dim,2)[:(dim // 2)].float() / dim))
    t =  torch.arange(end,device=freqs.device)
    freqs = torch.outer(t,freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs),freqs) # complex 64
    return freqs_cis


def apply_rotary_emb(x:torch.Tensor,freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1,2).float(),2,dim=-1),dim=-1))

    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1],x_out.shape[2],-1).transpose(1,2)

    return x_out


class Linear(nn.Module):
    def __init__(self,in_features:int,out_features: int,quant:bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features,in_features),dtype=torch.int8),
                requires_grad=False
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features,in_features)),
                requires_grad=False
            )
        self.quant = quant

    def forward(self,x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x,weight)
        return output


class embedding(nn.Module):

    def __int__(self,num_embeddings:int, embedding_dim: int,quant:bool):
        super().__int__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings,embedding_dim),dtype=torch.int8),
                requires_grad=False
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings,embedding_dim)),
                requires_grad=False
            )





