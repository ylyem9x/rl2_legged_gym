import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.managers.manager_term_cfg import *
from legged_gym.data import SimData, RobotData
from legged_gym.rl_env_cfg import ManagerBasedRLEnvCfg
from legged_gym.rl_env import ManagerBasedRLEnv
from legged_gym.envs.base.base import cfg


env = ManagerBasedRLEnv(cfg)

while True:
    env.step(torch.zeros(cfg.sim_data.num_envs, cfg.sim_data.num_actions,dtype=torch.float32, device=cfg.sim_data.device, requires_grad=False))