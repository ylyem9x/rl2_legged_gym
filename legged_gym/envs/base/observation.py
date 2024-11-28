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