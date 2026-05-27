import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import einops
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from pointnet2_ops import pointnet2_utils
from sat.sat_config.hyperparam import num_groups, drop_ratio


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


class MaskedGroup(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.num_group = num_groups
        self.group_size = group_size
        self.drop_ratio = drop_ratio

    def _drop_replace_idx(self, idx):
        """
        Vectorized drop-and-replace on idx.
        idx: [B, G, M] (long)
        """
        B, G, M = idx.shape
        k = int(self.drop_ratio * M)
        if k <= 0:
            return idx
        if k >= M:
            return idx

        device = idx.device

        perm = torch.rand(B, G, M, device=device).argsort(dim=-1)  # [B,G,M]
        drop_pos = perm[..., :k]          # [B,G,k]
        pool_pos = perm[..., k:]          # [B,G,M-k]

        choose = torch.randint(0, M - k, (B, G, k), device=device)  # [B,G,k]
        repl_pos = pool_pos.gather(-1, choose)  # [B,G,k]  

        values = idx.gather(-1, repl_pos)  # [B,G,k]

        idx_new = idx.clone()
        idx_new.scatter_(-1, drop_pos, values)

        return idx_new

    def forward(self, xyz, center=None, mask=None):
        """
        Inputs:
            xyz    : [B, N, 3]
            center : [B, G, 3]
            mask   : [B, N] bool, False not chosable
        Returns:
            neighborhood: [B, G, M, 3]
            center      : [B, G, 3]
        """

        if center is None:
            batch_size, num_points, _ = xyz.shape
            # fps the centers out
            center = fps(xyz, self.num_group) # B G 3
            # knn to get the neighborhood
            # _, idx = self.knn(xyz, center) # B G M
            idx = knn_point(self.group_size, xyz, center) # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            # normalize
            neighborhood = neighborhood - center.unsqueeze(2)
            return neighborhood, center
        else:
            B, N, _ = xyz.shape
            _, G, _ = center.shape
            M = self.group_size

            d2 = square_distance(center, xyz)  # [B, G, N]

            mask_expand = mask.unsqueeze(1).expand(B, G, N)
            d2 = d2.masked_fill(~mask_expand, float('inf'))

            _, idx = torch.topk(d2, k=M, dim=-1, largest=False, sorted=False)  # [B, G, M]

            # drop for augmentation
            idx = self._drop_replace_idx(idx)

            idx_base = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
            idx_flat = (idx + idx_base).view(-1)  # [B*G*M]
            xyz_flat = xyz.contiguous().view(B*N, 3)
            neigh = xyz_flat[idx_flat].view(B, G, M, 3)

            # normalize
            neigh = neigh - center.unsqueeze(2)  # [B, G, M, 3]

            return neigh, center


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointEncoder(nn.Module):
    def __init__(self,
                 num_groups: int=32,
                 group_size: int=16,
                 obs_horizon: int=2,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.obs_horizon = obs_horizon

        self.group_divider = Group(num_group = num_groups, group_size = group_size)

        self.group_embed = PointNetEncoderXYZ(
            in_channels=in_channels,
            out_channels=out_channels,
            use_layernorm=use_layernorm,
            final_norm=final_norm,
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )  

        self.token_attn = nn.MultiheadAttention(embed_dim=out_channels,
                                                num_heads=8,
                                                batch_first=True)
    
    def forward(self, x):
        B, T, N, _ = x.shape
        x = einops.rearrange(x, 'B T N A -> (B T) N A')
        neighborhood, center = self.group_divider(x) # (B T) G M 3 | (B T) G 3
        group_tokens = self.group_embed(neighborhood) # (B T) G F
        pos_embed = self.pos_embed(center) # (B T) G F
        tokens = group_tokens + pos_embed
        tokens = einops.rearrange(tokens, '(B T) G F -> B (T G) F', T=self.obs_horizon)

        if self.training:
            S = tokens.size(1)
            idx = torch.randperm(S, device=x.device)
            tokens = tokens[:, idx]

        tokens, _ = self.token_attn(tokens, tokens, tokens)  # B, S, F

        return tokens

    def seq_len(self):
        return self.num_groups * self.obs_horizon
        



class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
                
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
   
    def forward(self, x):
        x = self.mlp(x) # ..., N, f
        x = torch.max(x, -2)[0] # ..., f
        x = self.final_projection(x)
        return x

class PointNetDecoder(nn.Module):
    def __init__(self, point_size, latent_size, group_size, num_groups, obs_horizon):
        super(PointNetDecoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        self.obs_horizon = obs_horizon
        self.group_size = group_size
        self.num_groups = num_groups

        self.detokenizer = nn.Sequential(
            nn.Linear(latent_size, 64), nn.Mish(),
            nn.Linear(64, 128), nn.Mish(),
            nn.Linear(128, group_size*3),
            nn.Unflatten(-1, (group_size, 3))
        )
        self.group_merger = nn.Sequential(
            nn.Linear(num_groups*group_size, 512), nn.Mish(),
            nn.Linear(512, point_size)
        )
   
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.obs_horizon, self.point_size, 3)
    
    def forward(self, x):
        # x: B, seq_len, F
        B, S, F = x.shape
        x = einops.rearrange(x, 'B (T G) F -> B T G F', T=self.obs_horizon)
        x = self.detokenizer(x) # B T G M 3
        x = einops.rearrange(x, 'B T G M A -> B T A (G M)', G=self.num_groups)
        x = self.group_merger(x) # B T 3 point_size
        x = einops.rearrange(x, 'B T A N -> B T N A') # B, T, N, 3
        return x


