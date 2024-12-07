import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from .math import quat_apply_yaw

def get_body_indices(gym_env, name):
    sim, gym, robot_asset, envs, actor_handles = gym_env
    body_names = gym.get_asset_rigid_body_names(robot_asset)
    this_names = []
    for n in name:
        this_names.extend([s for s in body_names if n in s])
    indices = torch.zeros(len(this_names), dtype=torch.long, requires_grad=False)
    for i in range(len(this_names)):
            indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0],
                                                          this_names[i])
    return indices

def get_default_pos(gym_env, sim_data):
    sim, gym, robot_asset, envs, actor_handles = gym_env
    dof_names = gym.get_asset_dof_names(robot_asset)
    default_dof_pos = torch.zeros(sim_data.num_dofs, dtype=torch.float, device=sim_data.device, requires_grad=False)
    for i in range(sim_data.num_dofs):
        name = dof_names[i]
        angle = sim_data.init_state.default_joint_angles[name]
        default_dof_pos[i] = angle
    return default_dof_pos.unsqueeze(0)

def init_height_point(points_x, points_y, sim_data):
    """ Returns points at which the height measurments are sampled (in base frame)

    Returns:
        [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        [int]: number of elements
    """
    y = torch.tensor(points_y, device=sim_data.device, requires_grad=False)
    x = torch.tensor(points_x, device=sim_data.device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)

    num_height_points = grid_x.numel()
    points = torch.zeros(sim_data.num_envs, num_height_points, 3, device=sim_data.device, requires_grad=False)
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points, num_height_points

def get_height(sim_data, robot_data, height_point):
    """ Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    num_height_points = height_point[0, :, 0].numel()
    points = quat_apply_yaw(robot_data.base_quat.repeat(1, num_height_points), height_point) + (robot_data.root_state[:, :3]).unsqueeze(1)

    points += robot_data.terrain.cfg.border_size
    points = (points/robot_data.terrain.cfg.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, robot_data.height_sample.shape[0]-2)
    py = torch.clip(py, 0, robot_data.height_sample.shape[1]-2)

    heights1 = robot_data.height_sample[px, py]
    heights2 = robot_data.height_sample[px+1, py]
    heights3 = robot_data.height_sample[px, py+1]
    heights = torch.min(heights1, heights2)
    heights = torch.min(heights, heights3)

    return heights.view(sim_data.num_envs, -1) * robot_data.terrain.cfg.vertical_scale