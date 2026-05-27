from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import pi, nn, cat, stack, tensor, is_tensor
from torch.nn import Module, ModuleList
from torch.distributions import Normal
from torch.distributions.beta import Beta

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torch.utils.data import TensorDataset, DataLoader

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from hyper_connections import HyperConnections


LinearNoBias = partial(nn.Linear, bias = False)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def softclamp(t, value):
    if value <= 0.:
        return t

    return (t / value).tanh() * value

def softclamp_score_mod(value):
    def identity(score, b, h, q, k):
        return score

    def softclamped(score, b, h, q, k):
        score = score / value
        score = torch.tanh(score)
        score = score * value
        return score

    return softclamped if value > 0. else identity

def pad_at_dim(
    t,
    pad: tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

class MultiModalAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softclamp_value = 50.,
        learned_value_action_residual_mix = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = nn.RMSNorm(dim)

        # state parameters

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)


        # action parameters

        self.to_actions_qkvg = LinearNoBias(dim, 4 * dim_inner)

        self.to_action_value_residual_mix = nn.Sequential(
            LinearNoBias(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if learned_value_action_residual_mix else (lambda _: 0.5)

        self.to_actions_out = LinearNoBias(dim_inner, dim)

        # norms for all action linears
        # from Bytedance's GR-3

        self.softclamp_value = softclamp_value


    def forward(
        self,
        multimodal_seq,
        actions,
        return_keys_values = False,
        return_attn: bool = False,
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        multimodal_seq = self.rmsnorm(multimodal_seq)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av, ag = self.to_actions_qkvg(actions).chunk(4, dim = -1)

        mq, mk, mv, aq, ak, av, ag = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av, ag))

        q, k, v = tuple(cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))


        # attention
        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
        sim = softclamp(sim, self.softclamp_value)

        causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device)
        causal_mask[:seq_len, :seq_len] = False
        causal_mask[seq_len:, :] = False # actions have bidirectional attention, lining up with Transfusion paper
        causal_mask = causal_mask[None, None, ...]

        sim = sim.masked_fill(causal_mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # gating of values, used in alphafold line of work

        gates = pad_at_dim(ag.sigmoid(), (out.shape[-2] - ag.shape[-2], 0), value = 1., dim = -2)

        out = out * gates


        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        mout, aout = self.to_out(mout), self.to_actions_out(aout)

        output = (mout, aout)

        if return_attn and return_keys_values:
            return output, attn, (mk, mv, ak, av)
        if return_attn:
            return output, attn
        if return_keys_values:
            return output, (mk, mv, ak, av)
        return output



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.w = nn.Parameter(torch.Tensor(in_features, out_features)) 
        if bias:
            self.b = nn.Parameter(torch.Tensor(out_features)) 
        else:
            self.register_parameter('b', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5.0))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x: torch.Tensor, W_m: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_features)
        W_m: (B, in_features)
        """
        B, _ = x.shape
        w = (self.w).unsqueeze(0).repeat(B, 1, 1)
        W_m = W_m.unsqueeze(-1).repeat(1, 1, self.out_features)

        w_masked = w * W_m  # B, in_features, out_features
        output = einsum(w_masked, x, 'b i o, b i -> b o') + self.b # B, out_features
        
        return output





class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dim_inner = None,
        rmsnorm = True,
        norm_all = False
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim) if rmsnorm else nn.Identity()
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)

        # maybe additional norms for action branch

        self.post_proj_in_norm = nn.RMSNorm(dim_inner) if norm_all else nn.Identity()
        self.post_proj_out_norm = nn.RMSNorm(dim, elementwise_affine = False) if norm_all else nn.Identity()

    def forward(
        self,
        seq
    ):
        seq = self.rmsnorm(seq)
        seq, gates = self.proj_in(seq).chunk(2, dim = -1)

        seq = seq * F.gelu(gates)
        seq = self.post_proj_in_norm(seq)

        out = self.proj_out(seq)
        return self.post_proj_out_norm(out)

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    def forward(self, actions, cond):

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        normed = self.norm(actions)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return normed * gamma + beta


class AdaptiveLayerscale(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(self, actions, cond):

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        gamma = self.to_adaln_zero_gamma(cond)
        return actions * gamma.sigmoid()