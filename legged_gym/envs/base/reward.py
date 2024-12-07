import torch
import numpy as np
from legged_gym.data import SimData, RobotData

class basic_reward:
    def lin_vel_z(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize z axis base linear velocity
        return torch.square(robot_data.base_lin_vel[:, 2])

    def ang_vel_xy(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(robot_data.base_ang_vel[:, :2]), dim=1)

    def orientation(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize non flat base orientation
        return torch.sum(torch.square(robot_data.projected_gravity[:, :2]), dim=1)

    def torques(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize torques
        return torch.sum(torch.square(robot_data.torque), dim=1)

    def dof_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize dof velocities
        return torch.sum(torch.square(robot_data.dof_vel), dim=1)

    def dof_acc(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize dof accelerations
        return torch.sum(torch.square((robot_data.last_dof_vel - robot_data.dof_vel) / sim_data.dt), dim=1)

    def action_rate(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize changes in actions
        return torch.sum(torch.square(robot_data.last_action - robot_data.action), dim=1)

    def collision(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(robot_data.contact_force[:, robot_data.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def collision_force(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(robot_data.contact_force[:, robot_data.penalised_contact_indices, :], dim=-1))

    def termination(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Terminal reward / penalty
        return robot_data.reset_terminated

    # def dof_pos_limits(sim_data:SimData, robot_data:RobotData, cfg:dict):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(robot_data.dof_pos - robot_data.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    #     out_of_limits += (robot_data.dof_pos - robot_data.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def dof_vel_limits(sim_data:SimData, robot_data:RobotData, cfg:dict):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(robot_data.dof_vel) - robot_data.dof_vel_limits*robot_data.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def torque_limits(sim_data:SimData, robot_data:RobotData, cfg:dict):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(robot_data.torque) - robot_data.torque_limits*robot_data.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def tracking_lin_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(robot_data.command[:, :2] - robot_data.base_lin_vel[:, :2]), dim=1)
        sigma = cfg.get("sigma", 0.25)
        return torch.exp(-lin_vel_error/sigma)

    def tracking_ang_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(robot_data.command[:, 2] - robot_data.base_ang_vel[:, 2])
        sigma = cfg.get("sigma", 0.25)
        return torch.exp(-ang_vel_error/sigma)

    def feet_air_time(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        time_offset = cfg.get("time_offset", 0.5)
        contact_filt = torch.logical_or(robot_data.last_contact[:, :, 0], robot_data.last_contact[:, :, 1])
        robot_data.feet_air_time[robot_data.reset_env_ids] *= 0
        first_contact = (robot_data.feet_air_time > 0.) * contact_filt
        robot_data.feet_air_time += sim_data.dt
        rew_airTime = torch.sum((robot_data.feet_air_time - time_offset) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(robot_data.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        robot_data.feet_air_time *= ~contact_filt
        return rew_airTime

    def feet_slip(sim_data:SimData, robot_data:RobotData, cfg:dict):
        contact_filt = torch.logical_or(robot_data.last_contact[:, :, 0], robot_data.last_contact[:, :, 1]).squeeze(-1)
        foot_velocities = robot_data.rigid_body_state.view(sim_data.num_envs, sim_data.num_bodies, 13)[:, robot_data.feet_indices, 7:10]
        foot_velocities_square = torch.square(torch.norm(foot_velocities[:, :, 0:2], dim=2).view(sim_data.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities_square, dim=1)
        return rew_slip

    def stumble(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(robot_data.contact_force[:, robot_data.feet_indices, :2], dim=2) >\
                5 *torch.abs(robot_data.contact_force[:, robot_data.feet_indices, 2]), dim=1)

    def stand_still(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(robot_data.dof_pos - robot_data.default_dof_pos), dim=1) * (torch.norm(robot_data.command[:, :2], dim=1) < 0.1)

    def feet_contact_forces(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # penalize high contact forces
        num = cfg.get("max_contact_force", 100.)
        return torch.sum((torch.norm(robot_data.contact_force[:, robot_data.feet_indices, :], dim=-1) - num).clip(min=0.), dim=1)

    def joint_power(sim_data:SimData, robot_data:RobotData, cfg:dict):
        r = torch.clamp(robot_data.dof_vel * robot_data.torque, min=0.0)
        r = torch.sum(r, dim=1)
        return r

    def base_height(sim_data:SimData, robot_data:RobotData, cfg:dict):
        # Penalize base height away from target
        height_target = cfg.get("height_target", 0.4)
        base_height = torch.mean(robot_data.root_state[:, 2].unsqueeze(1) - robot_data.measured_height, dim=1)
        return torch.square(base_height - height_target)

    def live(sim_data:SimData, robot_data:RobotData, cfg:dict):
        return torch.ones_like(robot_data.reset_buf)