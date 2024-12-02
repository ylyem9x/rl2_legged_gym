import torch
import numpy as np
from legged_gym.data import SimData, RobotData
from legged_gym.utils.math import torch_rand

class vel3_command:
    dim = 3
    def resample(sim_data:SimData, robot_data:RobotData, command, cfg:dict):
        if robot_data.command_range == None:
            robot_data.command_range = []
            robot_data.command_range.append(cfg.get("command_range_x", (-1.0, 1.0)))
            robot_data.command_range.append(cfg.get("command_range_y", (-1.0, 1.0)))
            robot_data.command_range.append(cfg.get("command_range_yaw", (-1.0, 1.0)))
        interval = cfg.get("interval", 15) / sim_data.sim.dt
        env_ids = (robot_data.episode_length_buf % interval == 0).nonzero(as_tuple=False).squeeze(-1)
        for i in range(3):
            command[env_ids, i] = torch_rand(robot_data.command_range[i][0], robot_data.command_range[i][1],
                                            (len(env_ids), 1), device=sim_data.device).squeeze(1)
        offset = cfg.get("zero_offset", 0.2)
        command[env_ids, :2] *= (torch.norm(robot_data.command[env_ids, :2], dim=1) > offset).unsqueeze(1)
        return command

    def fixed_reset(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        return {}

    def curriculum_reset(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        """tracking_lin_vel reward is used to compute"""
        dv_per_env_finish = cfg.get("dv", 0.5)
        max_curriculum = cfg.get("max_curriculum", 2.0)
        percent = len(env_ids) / sim_data.num_envs
        if robot_data.extras["log"]["Episode_Reward/tracking_lin_vel"] > 0.8 * robot_data.reward_weight["tracking_lin_vel"]:
            robot_data.command_range[0][0] = np.clip(robot_data.command_range[0][0] - dv_per_env_finish * percent
                                                     , -max_curriculum, 0.)
            robot_data.command_range[0][1] = np.clip(robot_data.command_range[0][1] + dv_per_env_finish * percent
                                                     , 0., max_curriculum)
        return {"command_range_x":robot_data.command_range[0][0]}
