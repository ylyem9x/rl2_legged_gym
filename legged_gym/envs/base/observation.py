import torch
import numpy as np
from legged_gym.data import SimData, RobotData

class basic_obs:
    def base_ang_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.base_ang_vel

    def base_lin_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.base_lin_vel

    def projected_gravity(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.projected_gravity

    def command(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.command

    def dof_pos(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.dof_pos - robot_data.default_dof_pos

    def dof_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.dof_vel

    def action(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.action

    def p_gain(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.p_gain

    def d_gain(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.d_gain

    def motor_offset(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.motor_offset

    def motor_strength(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.motor_strength

    def base_mass(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.payloads

    def com_displacement(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.com_displacement

    def friction(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.friction

    def restitution(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return robot_data.restitution

    def obs_history(sim_data:SimData, robot_data:RobotData, cfg:dict):
        num_his = cfg.get("num_his", 1)
        num_obs = cfg.get("num_obs")
        if not hasattr(robot_data, "obs_his"):
            robot_data.obs_his = torch.zeros(sim_data.num_envs, num_his, num_obs, dtype=torch.float,
                                                device=sim_data.device, requires_grad=False)
        if "obs" in robot_data.obs:
            last_obs = robot_data.obs["obs"]
        else:
            last_obs = torch.zeros(sim_data.num_envs, num_obs, dtype=torch.float,
                                    device=sim_data.device, requires_grad=False)
        robot_data.obs_his = torch.cat((robot_data.obs_his[:, 1:], last_obs.unsqueeze(1)), dim = 1)
        reset_env_ids = robot_data.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            robot_data.obs_his[reset_env_ids, :] = 0
        return robot_data.obs_his

    def measure_height(sim_data:SimData, robot_data:RobotData, cfg:dict):
        offset = cfg.get("offset", 0.3)
        heights = robot_data.root_state[:, 2].unsqueeze(1) - robot_data.measured_height - offset
        return heights