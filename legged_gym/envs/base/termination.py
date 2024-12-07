import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.data import SimData, RobotData
from legged_gym.utils import init_height_point, get_height

class basic_termination:
    def time_out(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.episode_length_buf > sim_data.max_episode_length_s / sim_data.dt

    def random_time_out(sim_data:SimData, robot_data:RobotData, cfg:dict):
        probability = cfg.get("probability", 0.02) * sim_data.dt
        rand = torch.rand(sim_data.num_envs, dtype=float, requires_grad=False, device=sim_data.device)
        return rand < probability

    def contact(sim_data:SimData, robot_data:RobotData, cfg:dict):
        contact_offset = cfg.get("contact_offset", 1.)
        return torch.any(torch.norm(robot_data.contact_force[:, robot_data.termination_contact_indices, :], dim=-1) > contact_offset, dim=1)

    def base_height(sim_data:SimData, robot_data:RobotData, cfg:dict):
        """necessary when using trimesh terrain, beacuse robot may fall from edge."""
        terminal_body_height = cfg.get("terminal_height", 0.05)
        height_point, num = init_height_point([0.], [0.], sim_data)
        body_height = robot_data.root_state[:, 2] - get_height(sim_data, robot_data, height_point).squeeze(-1)
        return (body_height < terminal_body_height).squeeze(-1)