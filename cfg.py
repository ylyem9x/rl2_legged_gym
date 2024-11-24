import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.managers.manager_term_cfg import *
from legged_gym.data import SimData, RobotData
from legged_gym.rl_env_cfg import ManagerBasedRLEnvCfg
from legged_gym.rl_env import ManagerBasedRLEnv

def return0(sim_data:SimData, robot_data:RobotData, args):
    return torch.zeros(sim_data.num_envs,dtype=torch.float32, device=sim_data.device, requires_grad=False)
def returnFalse(sim_data:SimData, robot_data:RobotData, args):
    return torch.zeros(sim_data.num_envs,dtype=torch.bool, device=sim_data.device, requires_grad=False)


class reward:
    Reward_live = RewardTerm(return0, weight = 1.0)

class termination:
    time_out = TerminationTerm(returnFalse, time_out=True)

class action:
    actionscale = ActionSacleTerm(
        1.0 * np.ones(12), 1.0 * np.ones(12)
    )
    actioncompute = ActionComputeTerm(
        control_type="P", kp = 20.0, kd = 0.5
    )

cfg = ManagerBasedRLEnvCfg()
cfg.sim_data = SimData()
cfg.action = action()
cfg.reward = reward()
cfg.termination = termination()

env = ManagerBasedRLEnv(cfg)
while True:
    env.step(torch.zeros(cfg.sim_data.num_envs, cfg.sim_data.num_actions,dtype=torch.float32, device=cfg.sim_data.device, requires_grad=False))