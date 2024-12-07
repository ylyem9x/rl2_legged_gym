import time,json
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from DreamWaQ.algorithms import PPO
from DreamWaQ.modules import ActorCritic
from DreamWaQ.utils import VecEnv
from DreamWaQ.training_config import RunnerCfg

import matplotlib.pyplot as plt

class OnPolicyRunner:
    def __init__(
        self,
        env,
        cfg:RunnerCfg,# 六足的cfg不是RunnerCfg
        log_dir,
        device="cpu",
    ):
        self.train_cfg = cfg
        self.cfg = cfg.runner
        self.ppo_cfg = cfg
        self.policy_cfg = cfg.policy
        self.device = device
        self.env = env
        self.rew_his = []
        self.std_his = []
        self.est_loss = []
        self.terrain_level_his = []
        self.recon_loss = []
        self.kld_loss = []
        self.step =[]
        self.cv = 1
        self.env.num_obs = 45
        self.env.num_privileged_obs = 17*11 + 3
        self.env.num_actions = 12
        self.env.num_envs = 4096

        actor_critic = ActorCritic(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_actions,
            self.policy_cfg.num_latent,
            self.policy_cfg.num_history,
            self.policy_cfg.num_estimation,
            self.policy_cfg.activation,
            self.policy_cfg.actor_hidden_dims,
            self.policy_cfg.critic_hidden_dims,
            self.policy_cfg.encoder_hidden_dims,
            self.policy_cfg.decoder_hidden_dims,
        ).to(self.device)

        self.alg = PPO(
            actor_critic, cfg=self.ppo_cfg, device=self.device
        )
        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.policy_cfg.num_history, self.env.num_obs],
            [self.env.num_actions],
            self.policy_cfg.num_height_estimation
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._need_debug = False

        # if self.log_dir is not None:
        #     if not os.path.exists(self.log_dir):
        #         os.makedirs(self.log_dir)


    def learn(
        self,
        num_learning_iterations,
    ):
        self.save_cfg()
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(
                log_dir=self.log_dir, flush_secs=10
            )
        obs_dict = self.env.get_observations()
        for i, j in obs_dict.items():
            obs_dict[i] = j.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device,
        )

        tot_iter = (
            self.current_learning_iteration
            + num_learning_iterations
        )
        for it in range(
            self.current_learning_iteration, tot_iter
        ):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        obs_dict["obs"],
                        obs_dict["privileged_obs"],
                        obs_dict["obs_history"],
                        obs_dict["base_vel"],
                        self.cv
                    )
                    (
                        obs_dict,
                        rewards,
                        dones,
                        infos,
                    ) = self.env.step(actions)
                    for i, j in obs_dict.items():
                        obs_dict[i] = j.to(self.device)
                    rewards, dones = (
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(
                        rewards, dones, obs_dict["obs"], infos
                    )

                    if self.log_dir is not None:
                        # Book keeping
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(
                            as_tuple=False
                        )
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0]
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0]
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict["obs"],obs_dict["privileged_obs"],obs_dict["base_vel"])

            mean_value_loss, mean_surrogate_loss,mean_entropy_loss, mean_recons_loss, mean_est_loss, mean_kld_loss, mean_kl_div = self.alg.update(it)
            stop = time.time()
            learn_time = stop - start

            # cv but not applied yet
            # if len(rewbuffer) >= 2:
            #     self.cv = statistics.stdev(rewbuffer)/abs(statistics.mean(rewbuffer) + 1e-3)
            # print("cv:",self.cv)

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        "model_{}.pt".format(it),
                    )
                )
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir,
                "model_{}.pt".format(
                    self.current_learning_iteration
                ),
            )
        )

    def save_cfg(self):
        # 保存训练使用的cfg参数
        # env_cfg_dict = class_to_dict(self.env.cfg) if type(self.env.cfg) is not dict else self.env.cfg
        # train_cfg_dict = class_to_dict(self.train_cfg) if type(self.train_cfg) is not dict else self.train_cfg
        # env_cfg_json = json.dumps(env_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        # train_cfg_json = json.dumps(train_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        # with open(os.path.join(self.log_dir, "env_cfg.json"), 'w') as f:
        #     f.write(env_cfg_json)
        # with open(os.path.join(self.log_dir, "train_cfg.json"), 'w') as f:
        #     f.write(train_cfg_json)
        pass

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                if key == 'terrain_level':
                    terrain_level = value.item()
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/Reconstruction', locs['mean_recons_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation', locs['mean_est_loss'], locs['it'])
        self.writer.add_scalar('Loss/KL_div', locs['mean_kld_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/mean_kl_div', locs['mean_kl_div'], locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if self._need_debug:
            macro_score_string = ""
            for k,v in locs['macro_scores'].items():
                self.writer.add_scalar(f"Macro/{k}", v, locs['it'])
                macro_score_string += f"""{f'Mean macro {k}:':>{pad}} {v:.4f}\n"""
        else:
            macro_score_string = None
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Recons loss:':>{pad}} {locs['mean_recons_loss']:.4f}\n"""
                          f"""{'Est loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                          f"""{'KL loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Recons loss:':>{pad}} {locs['mean_recons_loss']:.4f}\n"""
                          f"""{'Est loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                          f"""{'KL loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += macro_score_string if macro_score_string is not None else ""

        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

        if len(locs['rewbuffer']) > 0:
            rew = statistics.mean(locs['rewbuffer'])
            self.std_his.append(mean_std.item())
            self.rew_his.append(rew)
            # self.terrain_level_his.append(terrain_level)
            self.step.append(locs['it'])
            self.recon_loss.append(locs['mean_recons_loss'])
            self.kld_loss.append(locs['mean_kld_loss'])
            self.est_loss.append(locs['mean_est_loss'])



    def hot_update(self):
        with open("hot_params/expert_hot.json") as f:
            hot_params = json.load(f)
        if hot_params['hot_update']:
            #! 仅支持 entropy ceof
            new_entropy_coef = hot_params['entropy_coef']
            self.alg.entropy_coef = new_entropy_coef
            print("Hot Update entropy: ",new_entropy_coef)
