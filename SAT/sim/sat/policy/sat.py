from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import os
import pytorch3d.ops as torch3d_ops

from sat.model.common.normalizer import LinearNormalizer
from sat.policy.base_policy import BasePolicy
from sat.common.pytorch_util import dict_apply
from sat.common.model_util import print_params
from sat.model.vision.obs_tokenizer import Obs_Tokenizer
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from sat.model.diffusion.sat import SAT
from sat.sat_config.joint_config import task_name_2_joint_desc
from sat.sat_config.hyperparam import num_groups, group_size, state_num_tokens, collect_attn, collect_sim

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, TargetConditionalFlowMatcher
from torchdyn.core import NeuralODE
import torchdiffeq

# Try to import FLOPs calculation libraries
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    cprint("[Warning] fvcore not available. Install with: pip install fvcore", "yellow")

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


class SATPolicy(BasePolicy):
    def __init__(self, 
            task_name: str,
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            fm_sigma=0.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.task_name = task_name

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = Obs_Tokenizer(observation_space=obs_dict,
                                    img_crop_shape=crop_shape,
                                    out_channel=encoder_output_dim,
                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                    use_pc_color=use_pc_color,
                                    pointnet_type=pointnet_type,
                                    )

        global_cond_dim = encoder_output_dim * n_obs_steps

                    

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SATPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SATPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = SAT(
            action_dim=action_dim,
            horizon=horizon,
            global_cond_dim=global_cond_dim,
            depth=8,
            num_heads=8
        )

        self.obs_encoder = obs_encoder
        self.model = model

        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.obs_dict = obs_dict
        self.encoder_output_dim = encoder_output_dim

        self.num_inference_steps = num_inference_steps

        self.flow_matching = TargetConditionalFlowMatcher(sigma=fm_sigma)

        print_params(self)



    # ################### flops calculation ###################
    def calculate_and_print_flops(self):
        """
        Calculate and print FLOPs, parameter count, and other model statistics
        """
        cprint("\n" + "="*80, "cyan")
        cprint("Model Computational Statistics", "cyan", attrs=["bold"])
        cprint("="*80, "cyan")
        
        # 1. Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.obs_encoder.parameters())
        model_params = sum(p.numel() for p in self.model.parameters())
        
        cprint(f"\n📊 Parameter Count:", "green", attrs=["bold"])
        cprint(f"  Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M)", "white")
        cprint(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)", "white")
        cprint(f"  Obs Encoder:          {encoder_params:,} ({encoder_params/1e6:.2f}M)", "white")
        cprint(f"  SAT Model:            {model_params:,} ({model_params/1e6:.2f}M)", "white")
        
        # 2. Memory estimation
        param_memory = total_params * 4 / (1024**2)  # FP32
        cprint(f"\n💾 Memory Estimation:", "green", attrs=["bold"])
        cprint(f"  Parameters (FP32):    {param_memory:.2f} MB", "white")
        cprint(f"  Parameters (FP16):    {param_memory/2:.2f} MB", "white")
        
        # 3. Calculate FLOPs
        try:
            flops_info = self._calculate_flops()
            if flops_info:
                cprint(f"\n⚡ FLOPs (Floating Point Operations):", "green", attrs=["bold"])
                for key, value in flops_info.items():
                    if isinstance(value, (int, float)):
                        if value >= 1e9:
                            cprint(f"  {key}: {value/1e9:.2f} GFLOPs", "white")
                        elif value >= 1e6:
                            cprint(f"  {key}: {value/1e6:.2f} MFLOPs", "white")
                        else:
                            cprint(f"  {key}: {value/1e3:.2f} KFLOPs", "white")
                    else:
                        cprint(f"  {key}: {value}", "white")
        except Exception as e:
            cprint(f"\n⚠️  FLOPs calculation failed: {str(e)}", "yellow")
        
        # 4. Model architecture info
        cprint(f"\n🏗️  Architecture Info:", "green", attrs=["bold"])
        cprint(f"  Action Dimension:     {self.action_dim}", "white")
        cprint(f"  Horizon:              {self.horizon}", "white")
        cprint(f"  Observation Steps:    {self.n_obs_steps}", "white")
        cprint(f"  Action Steps:         {self.n_action_steps}", "white")
        cprint(f"  Encoder Output Dim:   {self.encoder_output_dim}", "white")
        
        cprint("\n" + "="*80 + "\n", "cyan")
    
    def _calculate_flops(self):
        """
        Internal method to calculate FLOPs using available libraries
        """
        device = next(self.parameters()).device
        batch_size = 1
        
        # Create dummy inputs based on obs_dict
        dummy_obs = {}
        for key, shape in self.obs_dict.items():
            if key == 'point_cloud':
                # Point cloud: (B, T, N, C)
                n_points = shape[0] if len(shape) > 0 else 1024
                n_channels = shape[1] if len(shape) > 1 else 3
                if not self.use_pc_color:
                    n_channels = 3
                dummy_obs[key] = torch.randn(batch_size, self.n_obs_steps, n_points, n_channels).to(device)
            elif key == 'agent_pos':
                # Agent position: (B, T, D)
                dim = shape[0] if len(shape) > 0 else 10
                dummy_obs[key] = torch.randn(batch_size, self.n_obs_steps, dim).to(device)
        
        # Dummy inputs for SAT model
        dummy_trajectory = torch.randn(batch_size, self.horizon, self.action_dim).to(device)
        dummy_timestep = torch.randn(batch_size).to(device)
        
        # Joint description
        if self.task_name in task_name_2_joint_desc:
            joint_desc = task_name_2_joint_desc[self.task_name]
            dummy_joint_desc = torch.tensor(joint_desc).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            dummy_joint_desc = torch.randn(batch_size, self.action_dim, 2).to(device)
        
        flops_info = {}
        
        # Calculate encoder FLOPs
        if FVCORE_AVAILABLE:
            try:
                # Flatten observations for encoder
                dummy_obs_flat = dict_apply(dummy_obs, 
                    lambda x: x.reshape(-1, *x.shape[2:]))
                
                encoder_flops = FlopCountAnalysis(self.obs_encoder, (dummy_obs_flat,))
                encoder_total = encoder_flops.total()
                flops_info['Obs Encoder (per step)'] = encoder_total
                flops_info['Obs Encoder (total)'] = encoder_total * self.n_obs_steps
            except Exception as e:
                cprint(f"  Encoder FLOPs calculation failed: {e}", "yellow")
        
        # Calculate model FLOPs
        if FVCORE_AVAILABLE:
            try:
                # Get encoder output
                with torch.no_grad():
                    dummy_obs_flat = dict_apply(dummy_obs, 
                        lambda x: x.reshape(-1, *x.shape[2:]))
                    pts_token, state_token = self.obs_encoder(dummy_obs_flat)
                    pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=batch_size)
                    state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=batch_size)
                    dummy_cond = torch.cat([pts_token, state_token], dim=1)
                
                model_flops = FlopCountAnalysis(self.model, 
                    (dummy_trajectory, dummy_timestep, dummy_cond, dummy_joint_desc))
                model_total = model_flops.total()
                flops_info['SAT Model (per forward)'] = model_total
                
                if self.num_inference_steps:
                    flops_info['SAT Model (full inference)'] = model_total * self.num_inference_steps
                
                flops_info['Total (training forward)'] = flops_info.get('Obs Encoder (total)', 0) + model_total
                
            except Exception as e:
                cprint(f"  Model FLOPs calculation failed: {e}", "yellow")
        
        # Try thop if fvcore failed
        if not flops_info and THOP_AVAILABLE:
            try:
                dummy_obs_flat = dict_apply(dummy_obs, 
                    lambda x: x.reshape(-1, *x.shape[2:]))
                with torch.no_grad():
                    pts_token, state_token = self.obs_encoder(dummy_obs_flat)
                    pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=batch_size)
                    state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=batch_size)
                    dummy_cond = torch.cat([pts_token, state_token], dim=1)
                
                macs, params = profile(self.model, 
                    inputs=(dummy_trajectory, dummy_timestep, dummy_cond, dummy_joint_desc),
                    verbose=False)
                macs_str, params_str = clever_format([macs, params], "%.3f")
                
                flops_info['MACs (multiply-accumulate)'] = macs_str
                flops_info['Approximate FLOPs'] = macs * 2  # MACs * 2 ≈ FLOPs
                
            except Exception as e:
                cprint(f"  THOP calculation failed: {e}", "yellow")
        
        if not flops_info:
            # Manual estimation if libraries not available
            flops_info = self._estimate_flops_manually(batch_size)
        
        return flops_info
    
    def _calculate_flops(self):
        """
        Internal method to calculate FLOPs using available libraries
        """
        device = next(self.parameters()).device
        batch_size = 1
        
        # Create dummy inputs based on obs_dict
        dummy_obs = {}
        for key, shape in self.obs_dict.items():
            if key == 'point_cloud':
                # Point cloud: (B, T, N, C)
                n_points = shape[0] if len(shape) > 0 else 1024
                n_channels = shape[1] if len(shape) > 1 else 3
                if not self.use_pc_color:
                    n_channels = 3
                dummy_obs[key] = torch.randn(batch_size, self.n_obs_steps, n_points, n_channels).to(device)
            elif key == 'agent_pos':
                # Agent position: (B, T, D)
                dim = shape[0] if len(shape) > 0 else 10
                dummy_obs[key] = torch.randn(batch_size, self.n_obs_steps, dim).to(device)
        
        # Dummy inputs for SAT model
        dummy_trajectory = torch.randn(batch_size, self.horizon, self.action_dim).to(device)
        dummy_timestep = torch.randn(batch_size).to(device)
        
        # Joint description
        if self.task_name in task_name_2_joint_desc:
            joint_desc = task_name_2_joint_desc[self.task_name]
            dummy_joint_desc = torch.tensor(joint_desc).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            dummy_joint_desc = torch.randn(batch_size, self.action_dim, 2).to(device)
        
        flops_info = {}
        
        # Calculate encoder FLOPs
        if FVCORE_AVAILABLE:
            try:
                # Flatten observations for encoder
                dummy_obs_flat = dict_apply(dummy_obs, 
                    lambda x: x.reshape(-1, *x.shape[2:]))
                
                encoder_flops = FlopCountAnalysis(self.obs_encoder, (dummy_obs_flat,))
                encoder_total = encoder_flops.total()
                flops_info['Obs Encoder (per step)'] = encoder_total
                flops_info['Obs Encoder (total)'] = encoder_total * self.n_obs_steps
            except Exception as e:
                cprint(f"  Encoder FLOPs calculation failed: {e}", "yellow")
        
        # Calculate model FLOPs
        if FVCORE_AVAILABLE:
            try:
                # Get encoder output
                with torch.no_grad():
                    dummy_obs_flat = dict_apply(dummy_obs, 
                        lambda x: x.reshape(-1, *x.shape[2:]))
                    pts_token, state_token = self.obs_encoder(dummy_obs_flat)
                    pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=batch_size)
                    state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=batch_size)
                    dummy_cond = torch.cat([pts_token, state_token], dim=1)
                
                model_flops = FlopCountAnalysis(self.model, 
                    (dummy_trajectory, dummy_timestep, dummy_cond, dummy_joint_desc))
                model_total = model_flops.total()
                flops_info['SAT Model (per forward)'] = model_total
                
                if self.num_inference_steps:
                    flops_info['SAT Model (full inference)'] = model_total * self.num_inference_steps
                
                flops_info['Total (training forward)'] = flops_info.get('Obs Encoder (total)', 0) + model_total
                
            except Exception as e:
                cprint(f"  Model FLOPs calculation failed: {e}", "yellow")
        
        # Try thop if fvcore failed
        if not flops_info and THOP_AVAILABLE:
            try:
                dummy_obs_flat = dict_apply(dummy_obs, 
                    lambda x: x.reshape(-1, *x.shape[2:]))
                with torch.no_grad():
                    pts_token, state_token = self.obs_encoder(dummy_obs_flat)
                    pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=batch_size)
                    state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=batch_size)
                    dummy_cond = torch.cat([pts_token, state_token], dim=1)
                
                macs, params = profile(self.model, 
                    inputs=(dummy_trajectory, dummy_timestep, dummy_cond, dummy_joint_desc),
                    verbose=False)
                macs_str, params_str = clever_format([macs, params], "%.3f")
                
                flops_info['MACs (multiply-accumulate)'] = macs_str
                flops_info['Approximate FLOPs'] = macs * 2  # MACs * 2 ≈ FLOPs
                
            except Exception as e:
                cprint(f"  THOP calculation failed: {e}", "yellow")
        
        if not flops_info:
            # Manual estimation if libraries not available
            flops_info = self._estimate_flops_manually(batch_size)
        
        return flops_info
    
    def _estimate_flops_manually(self, batch_size=1):
        """
        Manual FLOPs estimation based on architecture
        """
        flops_info = {}
        
        # Estimate transformer FLOPs
        # For a transformer: FLOPs ≈ 12 * L * H^2 * S^2 per layer
        # where L = layers, H = hidden_dim, S = sequence_length
        
        try:
            # Get model config
            depth = 8  # from SAT initialization
            num_heads = 8
            # Assume hidden_dim from encoder_output_dim
            hidden_dim = self.encoder_output_dim
            seq_len = self.horizon  # action sequence length
            
            # Self-attention FLOPs per layer
            qkv_flops = 3 * seq_len * hidden_dim * hidden_dim
            attn_flops = 2 * num_heads * seq_len * seq_len * (hidden_dim // num_heads)
            output_flops = seq_len * hidden_dim * hidden_dim
            
            # FFN FLOPs per layer (typically 4x hidden_dim)
            ffn_flops = 2 * seq_len * hidden_dim * (4 * hidden_dim)
            
            # Total per layer
            layer_flops = qkv_flops + attn_flops + output_flops + ffn_flops
            
            # Total model FLOPs
            total_model_flops = depth * layer_flops * batch_size
            
            flops_info['Estimated SAT FLOPs'] = total_model_flops
            
            if self.num_inference_steps:
                flops_info['Estimated Full Inference'] = total_model_flops * self.num_inference_steps
            
            cprint("  Note: Using manual estimation (install fvcore for accurate FLOPs)", "yellow")
            
        except Exception as e:
            cprint(f"  Manual estimation failed: {e}", "yellow")
        
        return flops_info

        
    # ========= inference  ============
    def sample(self, x0, m, joint_desc):
        
        ts = torch.linspace(0., 1., self.num_inference_steps + 1).to(x0.device)
        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, t.unsqueeze(0), m, joint_desc),
            x0,
            ts,
            method="euler",
        )
        x1 = traj[-1]
        return x1


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], info=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'step' in obs_dict:
            del obs_dict['step']
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        joint_desc = task_name_2_joint_desc[self.task_name]
        joint_desc = torch.tensor(joint_desc).to(device).unsqueeze(0).repeat(B, 1, 1) # B, Da, 2

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            pts_token, state_token = self.obs_encoder(this_nobs) 
            ###############
            # pts_token = pts_token[:, 1:, :] ########### w.o. global token
            ###############
            pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=B)
            state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=B)
            m = torch.cat([pts_token, state_token], dim=1)

        x0 = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        # run sampling
        nsample = self.sample(x0, m, joint_desc)

        if collect_sim and B == 1:
            t0 = torch.tensor([0.]).float().to(x0.device)
            _ = self.model.forward(x0, t0, m, joint_desc, task_name=self.task_name)

        if collect_attn and B == 1:
            base_root = f"/aiarena/gpfs/vla/3D-Diffusion-Policy/vis_results/{self.task_name}"
            tname = datetime.now().strftime("%Y%m%d_%H%M%f")
            os.makedirs(base_root, exist_ok=True)
            t0 = torch.tensor([0.]).float().to(x0.device)
            _, attn_maps = self.model.forward(x0, t0, m, joint_desc, collect_attn=True)
            save_pth = os.path.join(base_root, tname + '.npy')
            np.save(save_pth, attn_maps)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def normalize_batch(self,  source_normalizer, \
        batch: Dict[str, Any], task_idx: torch.Tensor,
    ):
        batch_size = task_idx.shape[0]
        nobs = copy.deepcopy(batch['obs'])
        nactions = batch['action'] * 1.

        # group task by idx
        # print(task_idx)
        unique_tasks = torch.unique(task_idx)
        for t in unique_tasks.tolist():
            mask = (task_idx == int(t))
            if mask.sum() == 0:
                continue
            idx = mask.nonzero(as_tuple=True)[0]

            subbatch_nobs = {
                'agent_pos': batch['obs']['agent_pos'][idx],
                'point_cloud': batch['obs']['point_cloud'][idx],
            }
            subbatch_nactions = batch['action'][idx]

            subbatch_nobs_mask = {
                'agent_pos': batch['info']['mask_agent_pos'][idx],
                'point_cloud': batch['info']['mask_point_cloud'][idx],
            }
            subbatch_nactions_mask = batch['info']['mask_action'][idx]

            subbatch_nobs_original = {
                'agent_pos': (subbatch_nobs['agent_pos'] * 1.)[subbatch_nobs_mask['agent_pos']],
                'point_cloud': (subbatch_nobs['point_cloud'] * 1.)[subbatch_nobs_mask['point_cloud']],
            }
            subbatch_nactions_original = (subbatch_nactions * 1.)[subbatch_nactions_mask]

            normalized_sub_obs_original = source_normalizer[int(t)].normalize(subbatch_nobs_original)
            normalized_sub_action_original = source_normalizer[int(t)]['action'].normalize(subbatch_nactions_original)
            
            subbatch_nobs['agent_pos'][subbatch_nobs_mask['agent_pos']] = normalized_sub_obs_original['agent_pos']
            subbatch_nobs['point_cloud'][subbatch_nobs_mask['point_cloud']] = normalized_sub_obs_original['point_cloud']
            subbatch_nactions[subbatch_nactions_mask] = normalized_sub_action_original

            for k, v_norm in subbatch_nobs.items():
                nobs[k][idx] = v_norm
            nactions[idx] = subbatch_nactions

        return nobs, nactions



    def compute_loss(self, batch, source_normalizer=None):
        if source_normalizer is not None:
            task_idx = batch['info']['task_idx']
            nobs, nactions = self.normalize_batch(source_normalizer, batch, task_idx)
            mask_action = batch['info']['mask_action']
            mask_agent_pos = batch['info']['mask_agent_pos']
            mask_point_cloud = batch['info']['mask_point_cloud']
            center = batch['info']['down_pc']
            joint_desc = batch['info']['joint_desc'] # B, Da, 2
            if not self.use_pc_color:
                mask_point_cloud = mask_point_cloud[..., :3]
        else:
            # normalize input
            nobs = self.normalizer.normalize(batch['obs'])
            nactions = self.normalizer['action'].normalize(batch['action'])

            joint_desc = task_name_2_joint_desc[self.task_name]
            joint_desc = torch.tensor(joint_desc).to(nactions.device).unsqueeze(0).repeat(nactions.shape[0], 1, 1) # B, Da, 2


        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions      
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

            if source_normalizer is not None:
                this_mask_agent_pos = mask_agent_pos[:,:self.n_obs_steps,...].reshape(-1,*mask_agent_pos.shape[2:])
                this_mask_point_cloud = mask_point_cloud[:,:self.n_obs_steps,...].reshape(-1,*mask_point_cloud.shape[2:])
                this_center = center[:,:self.n_obs_steps,...].reshape(-1,*center.shape[2:])
                pts_token, state_token = self.obs_encoder(this_nobs, this_mask_agent_pos, this_mask_point_cloud, this_center)
            else:
                pts_token, state_token = self.obs_encoder(this_nobs)
            ###############
            # pts_token = pts_token[:, 1:, :] ########### w.o. global token
            ###############
            pts_token = rearrange(pts_token, '(B T) N F -> B N (T F)', B=batch_size)
            state_token = rearrange(state_token, '(B T) N F -> B N (T F)', B=batch_size)
            m = torch.cat([pts_token, state_token], dim=1)


        # Sample noise that we'll add to the images
        x0 = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps, xt, ut = self.flow_matching.sample_location_and_conditional_flow(x0, trajectory)
        
        vt = self.model(xt, 
                        timesteps, 
                        m,
                        joint_desc,
                        shuffle=True)

        loss = F.mse_loss(vt, ut, reduction='none')

        if source_normalizer is not None:
            loss = (loss * (mask_action.to(loss.dtype))).mean()
        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict

