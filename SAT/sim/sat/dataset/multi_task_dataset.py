from typing import Dict
import torch
import numpy as np
import copy
import fpsample
from sat.common.pytorch_util import dict_apply
from sat.common.replay_buffer import ReplayBuffer
from sat.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from sat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from sat.dataset.base_dataset import BaseDataset
from termcolor import cprint
from sat.sat_config.joint_config import task_name_2_joint_desc
from sat.sat_config.hyperparam import num_groups, group_size, dataset_ratio


def downsample_with_fps(points: np.ndarray, num_points: int = 512) -> np.ndarray:
    down_pts = []
    for t in range(points.shape[0]):
        sampled_indices = fpsample.bucket_fps_kdline_sampling(points[t, : ,:3], num_points, h=3)
        down_pts.append(points[t, sampled_indices])
    return np.stack(down_pts, axis=0)


def check_contiguous(data, down_point_cloud, joint_desc):
    """
    检查关键数组的连续性（contiguous）
    
    参数:
        data: 最终生成的data字典（包含obs、action、info等）
        down_point_cloud: 下采样后的点云数组
        joint_desc: 填充后的joint_desc数组
    """
    # 定义需要检查的数组及其名称（便于打印）
    arrays_to_check = [
        # 原始数据
        ("agent_pos (原始)", data['obs']['agent_pos']),  # 这里实际是padded后的，原始agent_pos需单独传参
        ("point_cloud (原始)", data['obs']['point_cloud']),  # 实际是padded后的
        ("action (原始)", data['action']),  # 实际是padded后的
        # 下采样点云
        ("down_point_cloud", down_point_cloud),
        # 填充后的数组
        ("agent_pos_padded", data['obs']['agent_pos']),
        ("mask_agent_pos", data['info']['mask_agent_pos']),
        ("pc_padded", data['obs']['point_cloud']),
        ("mask_point_cloud", data['info']['mask_point_cloud']),
        ("action_padded", data['action']),
        ("mask_action", data['info']['mask_action']),
        ("joint_desc (填充后)", joint_desc)
    ]
    
    # print("数组连续性检查结果：")
    # print("---------------------")
    for name, arr in arrays_to_check:
        # 检查是否为NumPy数组
        if not isinstance(arr, np.ndarray):
            print(f"{name}: 不是NumPy数组，无法检查")
            continue
        # 检查连续性
        is_contiguous = arr.flags.contiguous
        if not is_contiguous:
            print(f"{name}: {'连续' if is_contiguous else '不连续'} (形状: {arr.shape})")
    # print("---------------------")


class MultiTaskDataset(BaseDataset):
    def __init__(self,
            zarr_path_list, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name

        def get_task_name(s):
            after_slash = s.split('/')[-1]
            result = '_'.join(after_slash.split('_')[:2])
            return result

        self.task_name_list = [get_task_name(zarr_path) for zarr_path in zarr_path_list]
        cprint(self.task_name_list, color='red')
        # self.replay_buffer_list = [ReplayBuffer.copy_from_path( \
        #     zarr_path, keys=['state', 'action', 'point_cloud']) for zarr_path in zarr_path_list]
        self.replay_buffer_list = [ReplayBuffer.create_from_path( \
            zarr_path) for zarr_path in zarr_path_list]
        self.episode_len_list = [len(replay_buffer.episode_ends) for replay_buffer in self.replay_buffer_list]
        cprint(self.episode_len_list, color='red')

        self.num_task = len(self.replay_buffer_list)
        val_mask_list = [get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed) for replay_buffer in self.replay_buffer_list]
        rev_val_mask_list = [~val_mask for val_mask in val_mask_list]
        train_mask_list = [downsample_mask(
            mask=train_mask, 
            max_n=int(dataset_ratio * episode_len), 
            seed=seed) for train_mask, episode_len in zip(rev_val_mask_list, self.episode_len_list)]

        self.sampler_list = [SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask) for replay_buffer, train_mask in zip(self.replay_buffer_list, train_mask_list)]
        self.len_per_task = [len(samp) for samp in self.sampler_list]

        self.train_mask_list = train_mask_list
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.pts_num = 1024
        self.state_dim = 100
        self.action_dim = 100

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer_list = []

        num_samples = 10000

        for replay_buffer, task_name in zip(self.replay_buffer_list, self.task_name_list):

            total_samples = len(replay_buffer['action'])
            sample_size = min(total_samples, num_samples)
            indices = np.random.choice(total_samples, size=sample_size, replace=False)

            data = {
                'action': replay_buffer['action'][indices],
                'agent_pos': replay_buffer['state'][indices],
                # 'point_cloud': self.replay_buffer['point_cloud'],
                # 'hand_pts': self.replay_buffer['hand_pts'],
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            # normalizer['imagin_robot'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
            normalizer_list.append(normalizer)


        return normalizer_list

    def __len__(self) -> int:
        return sum(self.len_per_task)

    def find_index(self, x):
        """
        buckets: iterable of non-negative ints
        x: non-negative int
        """
        buckets = self.len_per_task
        if x < 0:
            raise ValueError("x must be non-negative")
        s = 0
        for i, b in enumerate(buckets):
            s += b
            if x < s:
                return i, x-(s-b)
        raise ValueError("x is >= sum(buckets)")

    def _pad_data(self, data, target_dim, pad_axis, fill_value=-100):
        current_shape = data.shape
        current_dim = current_shape[pad_axis]
        
        if current_dim >= target_dim:
            sliced_data = np.take(data, indices=range(target_dim), axis=pad_axis)
            mask = np.ones(list(sliced_data.shape[:pad_axis]) + [target_dim] + list(sliced_data.shape[pad_axis+1:]), dtype=bool)
            return sliced_data, mask
        else:
            pad_width = [(0, 0) for _ in current_shape]
            pad_width[pad_axis] = (0, target_dim - current_dim)
            padded_data = np.pad(data, pad_width, mode='constant', constant_values=fill_value)
            mask = np.zeros(padded_data.shape, dtype=bool)
            select_indices = [slice(None)] * padded_data.ndim
            select_indices[pad_axis] = slice(current_dim)
            mask[tuple(select_indices)] = True
            return padded_data, mask

    def _sample_to_data(self, sample, sampler_idx):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)[..., :3] # (T, 512, 3)
        action = sample['action'].astype(np.float32)

        down_point_cloud = downsample_with_fps(point_cloud[..., :3], num_points=num_groups)
        # down_point_cloud = np.ascontiguousarray(down_point_cloud)

        agent_pos_padded, mask_agent_pos = self._pad_data(
            data=agent_pos,
            target_dim=self.state_dim,
            pad_axis=-1,
        )
        pc_padded, mask_point_cloud = self._pad_data(
            data=point_cloud,
            target_dim=self.pts_num,
            pad_axis=-2,
        )
        action_padded, mask_action = self._pad_data(
            data=action,
            target_dim=self.action_dim,
            pad_axis=-1,
        )
        
        task_name = self.task_name_list[sampler_idx]
        joint_desc = np.array(task_name_2_joint_desc[task_name])
        joint_desc, _ = self._pad_data(
            data=joint_desc,
            target_dim=self.action_dim,
            pad_axis=0,
            fill_value=99,
        )

        info = {}

        info['task_idx'] = np.array(sampler_idx)
        
        info['joint_desc'] = joint_desc
        info['down_pc'] = down_point_cloud

        info['mask_action'] = mask_action  # (T, action_dim)
        info['mask_agent_pos'] = mask_agent_pos  # (T, state_dim)
        info['mask_point_cloud'] = mask_point_cloud  # (T, pts_num, 3)
        
        data = {
            'obs': {
                'point_cloud': pc_padded, # T, 512, 3
                'agent_pos': agent_pos_padded, # T, D_pos
            },
            'action': action_padded, # T, D_action
            'info': info,
        }

        # check_contiguous(
        #     data=data,
        #     down_point_cloud=down_point_cloud,
        #     joint_desc=joint_desc
        # )
    
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampler_idx, idx_offset = self.find_index(idx)
        sample = self.sampler_list[sampler_idx].sample_sequence(idx_offset)
        data = self._sample_to_data(sample, sampler_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
