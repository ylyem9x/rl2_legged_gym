import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.data import SimData, RobotData

class basic_termination:
    def time_out(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.episode_length_buf > sim_data.max_episode_length_s / sim_data.sim.dt

    def contact(sim_data:SimData, robot_data:RobotData, cfg:dict):
        contact_offset = cfg.get("contact_offset", 1.)
        return torch.any(torch.norm(robot_data.contact_force[:, robot_data.termination_contact_indices, :], dim=-1) > contact_offset, dim=1)