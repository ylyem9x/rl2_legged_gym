import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.utils.helpers import get_body_indices, get_default_pos

"""contain all tensor and array that need
"""

class SimData:
    """fixed data, never reset
    """
    class sim:
        """will be parse into gymapi.simParams()"""
        use_gpu_pipeline = True
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10., 0., 3.]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        env_spacing = 3.0 # only in plane
        # horizontal_scale = 0.1 # [m]
        # vertical_scale = 0.005 # [m]
        # border_size = 25 # [m]
        # curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # # rough terrain only:
        # measure_heights = True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # selected = False # select a unique terrain type and pass all arguments
        # terrain_kwargs = None # Dict of arguments for selected terrain
        # max_init_terrain_level = 5 # starting curriculum state
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows= 10 # number of terrain rows (levels)
        # num_cols = 20 # number of terrain cols (types)
        # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # # trimesh only:
        # slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset:
        file = "resources/robots/go1/urdf/go1_v2.urdf"
        name = "go1"  # actor name
        foot_name = "foot" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class init_state:
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
        }

    device = "cuda:0"
    sim_device_id = 0

    headless = False

    up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
    decimation = 4

    num_envs = 256
    num_bodies = None # read from urdf
    num_dofs = None # read from urdf
    num_actions = 12
    max_episode_length_s = 5



class RobotData:
    """changeable in sim
    """
    def __init__(self, sim_data: SimData, gym_tensor, gym_env, extra_gym_info):
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # update directly from gym
        self.root_state, self.dof_state, self.dof_pos, self.dof_vel, \
        self.base_quat, self.contact_force, self.rigid_body_state = gym_tensor
        self.gym_env = gym_env # (self.gym, self.robot_asset, self.envs, self.actor_handles)
        self.extra_gym_info = extra_gym_info #

        # fixed data
        self.gravity_vec = to_torch(get_axis_params(-1., sim_data.up_axis_idx), device=sim_data.device).repeat((sim_data.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=sim_data.device).repeat((sim_data.num_envs, 1))
        self.default_dof_pos = get_default_pos(gym_env, sim_data) # in size(1,num_dofs)
        self.env_origins = self.extra_gym_info["env_origins"] # in size(num_envs, 3)
        # get indices
        self.feet_indices = get_body_indices(gym_env, sim_data.asset.foot_name).to(sim_data.device)
        self.penalised_contact_indices = get_body_indices(gym_env, sim_data.asset.penalize_contacts_on).to(sim_data.device)
        self.termination_contact_indices = get_body_indices(gym_env, sim_data.asset.terminate_after_contacts_on).to(sim_data.device)

        # sim
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 7:10])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_state[:, 7:13])

        # log
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(sim_data.num_envs, dtype=torch.int, device=sim_data.device, requires_grad=False)
        self.extras = {}

        # action
        self.lag_buffer = None # available on pd/T control
        self.torque = torch.zeros(sim_data.num_envs, sim_data.num_actions, dtype=torch.float, device=sim_data.device, requires_grad=False)
        self.action = torch.zeros(sim_data.num_envs, sim_data.num_actions, dtype=torch.float, device=sim_data.device, requires_grad=False)
        self.last_action = torch.zeros(sim_data.num_envs, sim_data.num_actions, dtype=torch.float, device=sim_data.device, requires_grad=False)

        # reward
        self.reward_weight = dict()

        # command
        self.command = None
        self.command_range = None

        # # terrain
        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()
        # self.measured_heights = 0

        # reset
        self.reset_buf = None
        self.reset_terminated = None
        self.reset_time_out = None

        # domain random
        self.p_gain = 1.0
        self.d_gain = 1.0
        self.motor_offset = 0.0 # only on pd control
        self.motor_strength = 1.0
        self.noise_scale_vec = None

        # obs
        self.obs = None