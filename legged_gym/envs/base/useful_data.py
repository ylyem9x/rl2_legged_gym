import torch
import numpy as np
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import RobotDataTerm

class TwoFootContact:
    def init(sim_data:SimData, robot_data:RobotData):
        num_foot = robot_data.feet_indices.shape[0]
        contact = torch.zeros(sim_data.num_envs, num_foot, 2, device=sim_data.device, dtype=torch.bool)
        return contact

    def reset(sim_data:SimData, robot_data:RobotData, env_ids, last_contact):
        last_contact[env_ids, :, :] = False
        return {}

    def compute(sim_data:SimData, robot_data:RobotData, last_contact):
        contact = torch.norm(robot_data.contact_force[:, robot_data.feet_indices, 0:3], dim=2) > 1.
        last_contact = last_contact[:, :, 1:] + contact.unsqueeze(-1)

two_foot_contact = RobotDataTerm("last_contact", TwoFootContact.init, TwoFootContact.compute, TwoFootContact.reset)