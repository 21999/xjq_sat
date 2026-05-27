import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack
from torch.cuda.amp import autocast
import logging
from torch.nn.parallel import parallel_apply
import einops
from torch.jit import Final
from timm.layers import use_fused_attn
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from sat.model.common.siren_util import SineLayer, Sine
import pytorch_kinematics as pk


logger = logging.getLogger(__name__)


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def normal_modulate(x, scale, shift):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


def make_finger_mask():
    """
      - block indices:
          finger 1: rows&cols  6.. 9
          finger 2: rows&cols 10..13
          finger 3: rows&cols 14..17
          finger 4: rows&cols 18..21
    """
    D = 22
    mask = torch.zeros(D, D)

    mask[:, :6] = 1.0

    joints_per_finger = 4
    for f in range(4):
        start = 6 + f * joints_per_finger
        end = start + joints_per_finger
        for i in range(start, end):
            mask[i, start:(i + 1)] = 1.0

    return mask

class CausalAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            action_dim: int,
            horizon: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        mask_d = make_finger_mask()
        assert mask_d.shape == (action_dim, action_dim)
        self.register_buffer('mask_d', mask_d)  # (Da, Da)
        mask_t = torch.tril(torch.ones((horizon, horizon)))
        self.register_buffer('mask_t', mask_t)  # (T, T)
        mask_global = torch.kron(mask_t, mask_d).to(torch.bool)
        self.register_buffer('mask_global', mask_global) 
        self.action_dim = action_dim
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_global = self.mask_global
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            attn_mask = ~mask_global 
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.masked_fill(~mask_global.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CausalDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, action_dim, horizon, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalAttention(dim, action_dim, horizon, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class MaskAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            action_dim: int,
            horizon: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mask = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer('mask', mask.to(torch.bool))  # (Da, Da)
        self.action_dim = action_dim
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            attn_mask = ~mask 
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, action_dim, horizon, mlp_ratio=4.0, mask=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MaskAttention(dim, action_dim, horizon, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm, mask=mask)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class SinDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: Sine()
        self.mlp = nn.Sequential(
            SineLayer(dim, mlp_dim, is_first=True),
            SineLayer(mlp_dim, mlp_dim),
            nn.Linear(mlp_dim, dim)
        )
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class BiFinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        shift = shift.view(shift.size(0), 1, 1, -1)
        scale = scale.view(scale.size(0), 1, 1, -1)

        x = normal_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Mish(),
            nn.Linear(64, global_cond_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, global_cond_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(global_cond_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(global_cond_dim, action_dim)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.horizon)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, global_cond):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, action_dim)
        return x


class TokenDiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(),
            nn.Linear(64, global_cond_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon*action_dim, global_cond_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(global_cond_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(global_cond_dim, 1)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.horizon*self.action_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, global_cond):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """
        B, T, Da = x.shape
        x = x.flatten(1, 2)
        x = x.unsqueeze(-1) # B, T*Da, 1
        x = self.x_embedder(x) + self.pos_embed  # (N, T*Da, D)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T*Da, D)

        x = self.final_layer(x, c)                # (N, T*Da, 1)
        x = einops.rearrange(x, 'B (T Da) 1 -> B T Da', T=T)
        return x


class CausalTokenDiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.Mish(),
            nn.Linear(512, global_cond_dim*action_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon*action_dim, global_cond_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            CausalDiTBlock(global_cond_dim, num_heads, action_dim, horizon, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(global_cond_dim, 1)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.horizon*self.action_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, global_cond):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """
        # TODO：1. 从1map到embed dim其实不太对，因为joint和joint之间不一样，时间上可以一样  2. 树的权重 ！！！！！！！！！！！！！！！！！
        B, T, Da = x.shape

        # x = x.flatten(1, 2)
        # x = x.unsqueeze(-1) # B, T*Da, 1
        # x = self.x_embedder(x) + self.pos_embed  # (N, T*Da, D)

        x = self.x_embedder(x) # B, T, Da*D
        x = einops.rearrange(x, 'B T (Da D) -> B (T Da) D', Da=Da)
        x = x + self.pos_embed # (N, T*Da, D)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T*Da, D)

        x = self.final_layer(x, c)                # (N, T*Da, 1)
        x = einops.rearrange(x, 'B (T Da) 1 -> B T Da', T=T)
        return x


class BiTokenDiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.Mish(),
            nn.Linear(512, global_cond_dim*action_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.t_pos_embed = nn.Parameter(torch.zeros(1, horizon, global_cond_dim), requires_grad=True)
        self.a_pos_embed = nn.Parameter(torch.zeros(1, action_dim, global_cond_dim), requires_grad=True)

        mask_a = make_finger_mask()
        assert mask_a.shape == (action_dim, action_dim)
        mask_t = torch.tril(torch.ones((horizon, horizon)))

        self.depth = depth
        self.blocks_t = nn.ModuleList([
            MaskDiTBlock(global_cond_dim, num_heads, action_dim, horizon, mlp_ratio, mask=mask_t) for _ in range(depth)
        ])
        self.blocks_a = nn.ModuleList([
            MaskDiTBlock(global_cond_dim, num_heads, action_dim, horizon, mlp_ratio, mask=mask_a) for _ in range(depth)
        ])
        self.final_layer = BiFinalLayer(global_cond_dim, 1)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        t_pos_embed = get_1d_sincos_pos_embed(self.t_pos_embed.shape[-1], self.horizon)
        self.t_pos_embed.data.copy_(torch.from_numpy(t_pos_embed).float().unsqueeze(0))
        a_pos_embed = get_1d_sincos_pos_embed(self.a_pos_embed.shape[-1], self.action_dim)
        self.a_pos_embed.data.copy_(torch.from_numpy(a_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_t:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_a:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, global_cond):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """
        B, T, Da = x.shape

        x = self.x_embedder(x) # B, T, Da*D
        x = einops.rearrange(x, 'B T (Da D) -> B T Da D', Da=Da)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        c_t = einops.repeat(c, 'B D -> (B Da) D', Da=Da)
        c_a = repeat(c, 'B D -> (B T) D', T=T)

        for i in range(self.depth):
            x_t = einops.rearrange(x, 'B T Da D -> (B Da) T D')
            x_a = einops.rearrange(x, 'B T Da D -> (B T) Da D')
            block_t = self.blocks_t[i]
            block_a = self.blocks_a[i]
            x_t = x_t + self.t_pos_embed
            x_a = x_a + self.a_pos_embed
            x_t = block_t(x_t, c_t)  
            x_a = block_a(x_a, c_a)  
            x = einops.rearrange(x_t, '(B Da) T D -> B T Da D', B=B) + einops.rearrange(x_a, '(B T) Da D -> B T Da D', B=B)


        x = self.final_layer(x, c)                # (N, T, Da, 1)
        x = x.squeeze(-1)
        return x


class SinDiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            SineLayer(action_dim, 64, is_first=True),
            nn.Linear(64, global_cond_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, global_cond_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            SinDiTBlock(global_cond_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(global_cond_dim, action_dim)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.horizon)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, global_cond):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, action_dim)
        return x



class FKDiT(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        global_cond_dim,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.horizon = horizon

        self.x_embedder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Mish(),
            nn.Linear(64, global_cond_dim),
        )

        self.t_embedder = TimestepEmbedder(global_cond_dim)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, global_cond_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(global_cond_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(global_cond_dim, action_dim)

        self.final_layer_fk = FinalLayer(global_cond_dim, 24)

        self.initialize_weights()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.horizon)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # # Zero-out output layers:
        nn.init.constant_(self.final_layer_fk.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_fk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_fk.linear.weight, 0)
        nn.init.constant_(self.final_layer_fk.linear.bias, 0)


    def forward(self, x, t, global_cond, jac_embed):
        """
        Forward pass of DiT.
        x: (N, T, action_dim) tensor of action
        global_cond: (N, D) tensor of class labels
        """
        N, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        x = self.x_embedder(x) + self.pos_embed + jac_embed  # (N, T, D)

        t = self.t_embedder(t)                   # (N, D)
        c = global_cond + t                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x_o = self.final_layer(x, c)                # (N, T, action_dim)
        x_fk = self.final_layer_fk(x, c)                # (N, T, 24)

        return x_o, x_fk


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False, extra_tokens=0):
    """
    length
    return:
      pos_embed: [length, embed_dim] pr [extra_tokens+length, embed_dim]
    """
    pos = np.arange(length, dtype=np.float32)  # (length,)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)  # (length, embed_dim)
    if cls_token and extra_tokens > 0:
        zeros = np.zeros((extra_tokens, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([zeros, pos_embed], axis=0)  # add extra tokens
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 
    pos: 
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1.0 / (10000 ** omega)  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, embed_dim/2)

    emb_sin = np.sin(out)  # (M, embed_dim/2)
    emb_cos = np.cos(out)  # (M, embed_dim/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb