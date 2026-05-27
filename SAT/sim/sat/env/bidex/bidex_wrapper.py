import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import hydra

from termcolor import cprint
from gym import spaces


def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points


class BiDexEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task, rl_device, num_points=1024):

        self.task = task

        self.num_environments = task.num_envs # must be 1
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.clip_obs = 5.0
        self.clip_actions = 1.0
        self.rl_device = rl_device


        # self.env = task
        self.action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        self.num_points = num_points
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, 256, 256),
                dtype=np.float32
            ),    
            'agent_pos': spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'fk_state': spaces.Box(np.ones(62) * -np.Inf, np.ones(62) * np.Inf),
        })

        self.success = False

    def step(self, action: torch.Tensor):
        if action.ndim == 1:
            action = action.unsqueeze(0)

        actions_tensor = torch.clamp(action, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)
        st, reward, done, info = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras
        pc, fk_st, img = self.task.test_point_cloud_buf, self.task.test_state_buf, self.task.test_rgb_buf

        st = st[0]
        reward = reward[0]
        done = done[0]
        info = info['successes'][0]
        pc = pc[0]
        fk_st = fk_st[0]
        img = img[0]

        if info:
            self.success = True

        if pc.shape[0] > self.num_points:
            pc = downsample_with_fps(
                pc, self.num_points)

        if img.shape[0] != 3:  # make channel first
            img = img.permute(2, 0, 1)

        obs_dict = {
            'image': img,
            'agent_pos': st,
            'point_cloud': pc,
            'fk_state': fk_st,
        }
        return obs_dict, reward, done, {'success': self.success}

    def reset(self):
        self.success = False

        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))
        # step the simulator
        self.task.step(actions)

        st = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        pc, fk_st, img = self.task.test_point_cloud_buf, self.task.test_state_buf, self.task.test_rgb_buf

        st = st[0]
        pc = pc[0]
        fk_st = fk_st[0]
        img = img[0] / 255.

        if pc.shape[0] > self.num_points:
            pc = downsample_with_fps(
                pc, self.num_points)

        if img.shape[0] != 3:  # make channel first
            img = img.permute(2, 0, 1)

        obs_dict = {
            'image': img,
            'agent_pos': st,
            'point_cloud': pc,
            'fk_state': fk_st,
        }
        return obs_dict

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)


    def render(self, mode='rgb_array'):
        img = self.task.test_rgb_buf[0]
        if img.shape[0] == 3:  # make channel last
            img = img.permute(1, 2, 0)
        # to uint8
        img = img.cpu().numpy().astype(np.uint8)
        return img

    def close(self):
        pass

    def horizon(self):
        return self.env.horizon()

    def is_success(self):
        return self.success
