import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.managers.manager_term_cfg import *
from legged_gym.data import SimData, RobotData
from legged_gym.rl_env_cfg import ManagerBasedRLEnvCfg
from legged_gym.envs.base import *


class Reward:
    tracking_lin_vel = RewardTerm(basic_reward.tracking_lin_vel, 1.0, {"sigma":0.25})
    tracking_ang_vel = RewardTerm(basic_reward.tracking_ang_vel, 0.5, {"sigma":0.25})
    lin_vel_z = RewardTerm(basic_reward.lin_vel_z, -1.0)
    ang_vel_xy = RewardTerm(basic_reward.ang_vel_xy, -0.05)

ang_vel = ObservationTerm(basic_obs.base_ang_vel, noise=noise.uniform_noise(0.9, 1.1, "scale"), scale=0.25)
dof_pos = ObservationTerm(basic_obs.dof_pos, noise=noise.uniform_noise(0.9, 1.1, "scale"), scale=1.0)
lin_vel = ObservationTerm(basic_obs.base_lin_vel, noise=noise.uniform_noise(0.9, 1.1, "scale"), scale=0.25)
cmd = ObservationTerm(basic_obs.command, scale = 1.0)
class Observation:
    obs = ObservationGroup(ang_vel, cmd)
    priv_obs = ObservationGroup(lin_vel)

class Event:
    reset_dof = EventTerm(basic_event.reset_dof, mode="reset", cfg={"limit":(0.5, 1.5)})
    reset_root_states = EventTerm(basic_event.reset_root_states, mode="reset")
    random_base = EventTerm(basic_event.random_base_props, mode="startup", cfg={"payload":(-1.0,1.0),"com_displacement":(-0.01,0.01)})
    random_rigid = EventTerm(basic_event.random_rigid_props, mode="startup", cfg={"friction":(0.3,3.0),"restitution":(0.0,0.2)})

class Termination:
    time_out = TerminationTerm(basic_termination.time_out,time_out=True)
    contact = TerminationTerm(basic_termination.contact, time_out=False, cfg={"contact_offset":100.})

class Action:
    actionscale = ActionSacleTerm(1.0,100.0)
    actioncompute = ActionComputeTerm(control_type="P", kp = 20.0, kd = 0.5)

class Command:
    cmd = CommandTerm(vel3_command.dim, vel3_command.resample, vel3_command.curriculum_reset,cfg={
        "interval": 2,
        "command_range_x": (-1.0, 1.0),
        "command_range_y": (-1.0, 1.0),
        "command_range_yaw": (-1.0, 1.0),
        "dv": 0.2
    })

cfg = ManagerBasedRLEnvCfg()
cfg.reward = Reward()
cfg.obs = Observation()
cfg.event = Event()
cfg.termination = Termination()
cfg.action = Action()
cfg.command = Command()
