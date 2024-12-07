import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.managers.manager_term_cfg import *
from legged_gym.data import SimData, RobotData
from legged_gym.rl_env_cfg import ManagerBasedRLEnvCfg
from legged_gym.envs.base import *
from legged_gym.utils.terrain import Terrain

sim_data = SimData()
sim_data.terrain.mesh_type = "trimesh"
class Reward:
    # live = RewardTerm(basic_reward.live, 0.5)
    tracking_lin_vel = RewardTerm(basic_reward.tracking_lin_vel, 1.5, {"sigma":0.25})
    tracking_ang_vel = RewardTerm(basic_reward.tracking_ang_vel, 0.5, {"sigma":0.25})
    lin_vel_z = RewardTerm(basic_reward.lin_vel_z, -2.0)
    ang_vel_xy = RewardTerm(basic_reward.ang_vel_xy, -0.05)
    orientation = RewardTerm(basic_reward.orientation, -0.2)
    dof_acc = RewardTerm(basic_reward.dof_acc, -1e-7)
    joint_power = RewardTerm(basic_reward.joint_power, -2e-5)
    base_height = RewardTerm(basic_reward.base_height, -1.0, {"height_target": 0.35})
    action_rate = RewardTerm(basic_reward.action_rate, -0.01)
    # foot_slip = RewardTerm(basic_reward.feet_slip, weight=-0.001)
    # collision = RewardTerm(basic_reward.collision, -0.02)

ang_vel = ObservationTerm(basic_obs.base_ang_vel, noise = noise.uniform_noise(), scale=0.25)
dof_pos = ObservationTerm(basic_obs.dof_pos, noise = noise.uniform_noise(), scale=1.0)
dof_vel = ObservationTerm(basic_obs.dof_vel, noise = noise.uniform_noise(), scale=0.05)
action = ObservationTerm(basic_obs.action, scale=1.0)
proj_gravity = ObservationTerm(basic_obs.projected_gravity, noise = noise.uniform_noise(), scale=1.0)
lin_vel = ObservationTerm(basic_obs.base_lin_vel, scale=1.0)
cmd = ObservationTerm(basic_obs.command, scale = torch.tensor([2.0, 2.0, 0.25], device="cuda:0"))
obs_his = ObservationTerm(basic_obs.obs_history, scale = 1.0, cfg={"num_his":6,"num_obs":45})
measure_height = ObservationTerm(basic_obs.measure_height, scale=5.0)
class Observation:
    obs = ObservationGroup(dof_pos, dof_vel, ang_vel, cmd, proj_gravity, action)
    privileged_obs = ObservationGroup(lin_vel, measure_height)
    obs_history = ObservationGroup(obs_his)
    base_vel = ObservationGroup(lin_vel)

class Event:
    random_base = EventTerm(basic_event.random_base_props, mode="startup", cfg={"payload":(-1.0,1.0),"com_displacement":(-0.01,0.01)})
    random_rigid = EventTerm(basic_event.random_rigid_props, mode="startup", cfg={"friction":(0.3,3.0),"restitution":(0.0,0.2)})

    a_terrain_curriculum = EventTerm(basic_event.terrain_curriculum, mode="reset")
    reset_dof = EventTerm(basic_event.reset_dof, mode="reset", cfg={"limit":(0.5, 1.5)})
    reset_root_states = EventTerm(basic_event.reset_root_states, mode="reset")
    random_pd = EventTerm(basic_event.random_PD_gain, mode="reset")
    random_motor = EventTerm(basic_event.random_motor_strength, mode="reset")
    random_pos = EventTerm(basic_event.random_motor_offset, mode="reset", cfg={"offset":[-0.1,0.1]})

    # push_vel = EventTerm(basic_event.random_changing_base_vel, mode="interval", interval={"global":15.}, cfg={"max":1.0})

class Termination:
    time_out = TerminationTerm(basic_termination.time_out,time_out=True)
    random = TerminationTerm(basic_termination.random_time_out, time_out=True, cfg={"probability":0.01})
    contact = TerminationTerm(basic_termination.contact, time_out=False, cfg={"contact_offset":10.})
    termination_height = TerminationTerm(basic_termination.base_height, time_out=False)

class Action:
    actionscale = ActionSacleTerm(0.25,10.0)
    actioncompute = ActionComputeTerm(control_type="P", kp = 20.0, kd = 0.5)

class Command:
    cmd = CommandTerm(vel3_command.dim, vel3_command.resample, vel3_command.curriculum_reset,cfg={
        "interval": 15,
        "command_range_x": [-1.0, 1.0],
        "command_range_y": [-0.6, 0.6],
        "command_range_yaw": [-0.5, 0.5]
    })

cfg = ManagerBasedRLEnvCfg()
cfg.robot_data_terms.append(two_foot_contact)
cfg.sim_data = sim_data
cfg.reward = Reward()
cfg.obs = Observation()
cfg.event = Event()
cfg.termination = Termination()
cfg.action = Action()
cfg.command = Command()
cfg.terrain = Terrain(sim_data.terrain, sim_data.num_envs)
