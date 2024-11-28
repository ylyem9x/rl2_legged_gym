import torch
import numpy as np
from legged_gym.data import SimData, RobotData

class basic_reward:
    def lin_vel_z(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize z axis base linear velocity
        return torch.square(robot_data.base_lin_vel[:, 2])

    def ang_vel_xy(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(robot_data.base_ang_vel[:, :2]), dim=1)

    def orientation(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize non flat base orientation
        return torch.sum(torch.square(robot_data.projected_gravity[:, :2]), dim=1)

    # def base_height(sim_data:SimData, robot_data:RobotData, cfg):
    #     # Penalize base height away from target
    #     base_height = torch.mean(robot_data.root_states[:, 2].unsqueeze(1) - robot_data.measured_heights, dim=1)
    #     return torch.square(base_height - robot_data.cfg.rewards.base_height_target)

    def torques(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize torques
        return torch.sum(torch.square(robot_data.torque), dim=1)

    def dof_vel(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize dof velocities
        return torch.sum(torch.square(robot_data.dof_vel), dim=1)

    def dof_acc(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize dof accelerations
        return torch.sum(torch.square((robot_data.last_dof_vel - robot_data.dof_vel) / sim_data.sim.dt), dim=1)

    def action_rate(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize changes in actions
        return torch.sum(torch.square(robot_data.last_action - robot_data.action), dim=1)

    def collision(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(robot_data.contact_force[:, robot_data.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def termination(sim_data:SimData, robot_data:RobotData, cfg):
        # Terminal reward / penalty
        return robot_data.reset_terminated

    # def dof_pos_limits(sim_data:SimData, robot_data:RobotData, cfg):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(robot_data.dof_pos - robot_data.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    #     out_of_limits += (robot_data.dof_pos - robot_data.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def dof_vel_limits(sim_data:SimData, robot_data:RobotData, cfg):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(robot_data.dof_vel) - robot_data.dof_vel_limits*robot_data.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def torque_limits(sim_data:SimData, robot_data:RobotData, cfg):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(robot_data.torque) - robot_data.torque_limits*robot_data.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def tracking_lin_vel(sim_data:SimData, robot_data:RobotData, cfg):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(robot_data.command[:, :2] - robot_data.base_lin_vel[:, :2]), dim=1)
        sigma = cfg.get("sigma", 0.25)
        return torch.exp(-lin_vel_error/sigma)

    def tracking_ang_vel(sim_data:SimData, robot_data:RobotData, cfg):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(robot_data.command[:, 2] - robot_data.base_ang_vel[:, 2])
        sigma = cfg.get("sigma", 0.25)
        return torch.exp(-ang_vel_error/sigma)

    # def feet_air_time(sim_data:SimData, robot_data:RobotData, cfg):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = robot_data.contact_force[:, robot_data.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, robot_data.last_contact)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    def stumble(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(robot_data.contact_force[:, robot_data.feet_indices, :2], dim=2) >\
                5 *torch.abs(robot_data.contact_force[:, robot_data.feet_indices, 2]), dim=1)

    def stand_still(sim_data:SimData, robot_data:RobotData, cfg):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(robot_data.dof_pos - robot_data.default_dof_pos), dim=1) * (torch.norm(robot_data.command[:, :2], dim=1) < 0.1)

    def feet_contact_forces(sim_data:SimData, robot_data:RobotData, cfg):
        # penalize high contact forces
        num = cfg.get("max_contact_force", 0.25)
        return torch.sum((torch.norm(robot_data.contact_force[:, robot_data.feet_indices, :], dim=-1) - num).clip(min=0.), dim=1)