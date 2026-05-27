import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from sat.env import BiDexEnv
from sat.gym_util.multistep_wrapper import MultiStepWrapper
from sat.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from sat.policy.base_policy import BasePolicy
from sat.common.pytorch_util import dict_apply
from sat.env_runner.base_runner import BaseRunner
import sat.common.logger_util as logger_util

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.tasks.shadow_hand_over import ShadowHandOver


class BiDexRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        import sys
        import os
        _orig_argv = sys.argv.copy()
        _orig_cwd  = os.getcwd()
        # copy orginal
        sys.argv = [
            "train.py",  
            f"--task={task_name}",
            "--algo=ppo",
            "--test",
        ]
        os.chdir("/aiarena/gpfs/vla/3D-Diffusion-Policy/third_party/DexterousHands/bidexhands")
        args = get_args()
        # args.task = task_name
        # args.algo = 'ppo'
        # args.test = True
        device_id = args.device_id
        rl_device = args.rl_device
        is_test = args.test
        cfg, cfg_train, logdir = load_cfg(args)
        sim_params = parse_sim_params(args, cfg, cfg_train)

        cprint(f'{task_name}: is test: {is_test}', 'red')

        task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False,
                is_test=is_test,)

        sys.argv = _orig_argv
        os.chdir(_orig_cwd)

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(BiDexEnv(
                    task,
                    rl_device=rl_device
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn()
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy, vqvae=None):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []

        
        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()
            policy.reset()

            done = False
            reward_sum = 0.
            actual_step_count = 0
            while not done:

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs['agent_pos'].unsqueeze(0)
                    obs_dict_input['fk_state'] = obs['fk_state'].unsqueeze(0)
                    obs_dict_input['step'] = actual_step_count
                    if vqvae is None:
                        action_dict = policy.predict_action(obs_dict_input)
                    else:
                        action_dict = policy.predict_action(obs_dict_input, vqvae)

                # step env
                action = action_dict['action'].squeeze(0)
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward.item()
                actual_step_count += 1

                if done:
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())

       

        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train

        log_data['test_mean_score'] = SR_mean_train

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {SR_mean_train:.3f}", 'green')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data
