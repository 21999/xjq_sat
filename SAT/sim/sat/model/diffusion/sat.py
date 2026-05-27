import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention
from sat.model.common.masked_attn import MultiModalAttention
import torch.nn.functional as F
from einops import repeat, pack, unpack, rearrange
from torch.cuda.amp import autocast
import logging
from sat.sat_config.hyperparam import collect_sim
import os
from datetime import datetime


logger = logging.getLogger(__name__)


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        self.attn = MultiModalAttention(dim, heads=num_heads)
        # flasth attn can not be used with jvp
        # self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, m, a, c, return_attn: bool = False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        if return_attn:
            out_tuple, attn_map = self.attn(m, modulate(self.norm1(a), scale_msa, shift_msa), return_attn=True)
            a = a + gate_msa.unsqueeze(1) * out_tuple[1]
        else:
            a = a + gate_msa.unsqueeze(1) * self.attn(
                m, modulate(self.norm1(a), scale_msa, shift_msa)
            )[1]
        a = a + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(a), scale_mlp, shift_mlp)
        )
        if return_attn:
            return m, a, attn_map
        return m, a


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


class SAT(nn.Module):
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
        max_seq_len = 100 
        # max_seq_len = action_dim

        self.num_heads = num_heads
        self.seq_len = max_seq_len
        self.horizon = horizon
        self.action_dim = action_dim
        self.non_shuffle_idx = [ i for i in range(action_dim)]

        self.x_embedder = nn.Sequential(
            nn.Linear(horizon, 64),
            nn.Mish(),
            nn.Linear(64, global_cond_dim),
        )
        self.t_embedder = TimestepEmbedder(global_cond_dim * 2)

        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, global_cond_dim), requires_grad=True)

        self.robot_embed = nn.Embedding(100, global_cond_dim // 2)
        self.joint_embed = nn.Embedding(100, global_cond_dim // 2)

        self.blocks = nn.ModuleList([
            DiTBlock(global_cond_dim * 2, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(2*global_cond_dim, horizon)

        # self.shuffle_embedder = nn.Embedding(action_dim, global_cond_dim)

        self.global_cond_dim = global_cond_dim

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

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_len)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
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


    def forward(self, a, t, m, joint_desc, shuffle=False, collect_attn=False, task_name='test'):
        """
        Forward pass of DiT.
        a: (N, T, action_dim) tensor of action
        joint_desc: (B, Da, 2)
        t: (N,) tensor of diffusion timesteps t
        y: (N,) tensor of class labels
        """
        B, T, Da = a.shape
        _, m_seq_len, _ = m.shape

        a = rearrange(a, 'B T Da -> B Da T')

        shuffle_idx = None
        if shuffle:
            shuffle_idx = torch.cat([torch.randperm(Da, device=a.device).unsqueeze(0) for _ in range(B)], dim=0)
            # shuffle_idx = torch.stack([torch.randperm(Da, device=x.device) for _ in range(B)], dim=0)

            shuffle_idx_expanded = shuffle_idx.unsqueeze(-1).repeat(1, 1, T)  # (B, Da, T)
            a = torch.gather(a, dim=1, index=shuffle_idx_expanded)  # (B, Da, T)


        robot_embeds = self.robot_embed(joint_desc[..., 0]) # (B, Da, F/2)
        joint_embeds = self.joint_embed(joint_desc[..., 1]) # (B, Da, F/2)
        pos_embed_expanded = torch.cat([robot_embeds, joint_embeds], dim=-1)

        if B == 1 and collect_sim:
            base_root = f"/aiarena/gpfs/vla/3D-Diffusion-Policy/vis_results/{task_name}"
            tname = datetime.now().strftime("%Y%m%d_%H%M%f")

            je = joint_embeds.squeeze(0)  # torch.Tensor, (Da, F)
            je = je.detach()  
            je_cpu = je.cpu().numpy()
            je_norm = F.normalize(je, dim=1)             # (Da, F)
            sim = (je_norm @ je_norm.t()).cpu().numpy()  # (Da, Da)

            os.makedirs(base_root, exist_ok=True)
            np.save(os.path.join(base_root, f"joint_embeds_{tname}.npy"), je_cpu)
            np.save(os.path.join(base_root, f"joint_sim_{tname}.npy"), sim)
            np.savetxt(os.path.join(base_root, f"joint_sim_{tname}.csv"), sim, delimiter=",")


        if shuffle:
            # pos_embed_expanded = self.pos_embed.expand(B, -1, -1)  # (B, Da, F)
            pos_embed_shuffled = torch.gather(pos_embed_expanded, dim=1, index=shuffle_idx.unsqueeze(-1).repeat(1, 1, self.global_cond_dim))  # (B, Da, F)
            a = torch.cat([self.x_embedder(a), pos_embed_shuffled], dim=-1)   # (B, Da, 2F)
        else:
            if a.shape[1] == self.seq_len:
                a = torch.cat([self.x_embedder(a), pos_embed_expanded], dim=-1)  # (B, Da, 2F) 
            else:
                a = torch.cat([self.x_embedder(a), pos_embed_expanded[:, self.non_shuffle_idx, :]], dim=-1)  # (B, Da, 2F) 
            

        c = self.t_embedder(t)                   # (N, 2F)

        attn_maps = [] if collect_attn else None
        for i, block in enumerate(self.blocks):
            if collect_attn:
                m, a, attn = block(m, a, c, return_attn=True)
                attn_maps.append(attn.detach().cpu())
            else:
                m, a = block(m, a, c)                      # (N, D, 2F)

        a = self.final_layer(a, c)                # (N, D, T)
        a = rearrange(a, 'B Da T -> B T Da')

        if shuffle:
            inv_shuffle_idx = torch.argsort(shuffle_idx, dim=1)  # (B, Da)
            inv_shuffle_idx_expanded = inv_shuffle_idx.unsqueeze(1).repeat(1, T, 1)  # (B, T, Da)
            x_pred_restored = torch.gather(a, dim=2, index=inv_shuffle_idx_expanded) 

            if collect_attn:
                return x_pred_restored, attn_maps
            return x_pred_restored
        else:
            if collect_attn:
                return a, attn_maps
            return a


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