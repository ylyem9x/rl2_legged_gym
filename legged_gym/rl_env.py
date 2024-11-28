from time import time
from warnings import WarningMessage
import numpy as np
import os
import sys

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.data import SimData, RobotData
from legged_gym.rl_env_cfg import ManagerBasedRLEnvCfg
# from legged_gym.utils.terrain import Terrain
from legged_gym.managers import RewardManager, TerminationManager, CommandManager, \
                                EventManager, ObservationManager, ActionManager, CurriculumManager
from legged_gym.utils.helpers import class_to_dict

class ManagerBasedRLEnv:
    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        self.cfg = cfg
        self.sim_data = cfg.sim_data
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.enable_viewer_sync = True
        self.viewer = None
        self.gym.prepare_sim(self.sim)
        self.prepare_viewer_keyboard()
        if not self.sim_data.headless:
            self._set_camera(self.sim_data.viewer.pos, self.sim_data.viewer.lookat)

        self._init_buffers()
        self._load_manager()

    def create_sim(self):
        sim_cfg = {"sim":class_to_dict(self.sim_data.sim)}
        self.sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(sim_cfg["sim"], self.sim_params)
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_data.device)
        if self.sim_data.headless:
            self.graphics_device_id = -1
        else:
            self.graphics_device_id = self.sim_device_id
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)
        mesh_type = self.sim_data.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # self.terrain = Terrain(self.sim_data.terrain, self.sim_data.num_envs)
            pass
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        self._create_envs()

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_state = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.sim_data.num_envs, self.sim_data.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.sim_data.num_envs, self.sim_data.num_dofs, 2)[..., 1]
        self.base_quat = self.root_state[:, 3:7]

        self.contact_force = gymtorch.wrap_tensor(net_contact_force).view(self.sim_data.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.sim_data.num_envs * self.num_bodies, :]

        gym_tensor = (self.root_state, self.dof_state, self.dof_pos, self.dof_vel, self.base_quat, self.contact_force, self.rigid_body_state)
        gym_env = (self.gym, self.robot_asset, self.envs, self.actor_handles)
        extra_gym_info = dict()
        extra_gym_info["base_init_state"] = self.base_init_state
        extra_gym_info["env_origins"] = self.env_origins
        self.robot_data = RobotData(self.sim_data, gym_tensor, gym_env, extra_gym_info)

    def _load_manager(self):
        self.termination_manager = TerminationManager(self.cfg.termination, self.sim_data, self.robot_data)
        print("[INFO] termination Manager: ", self.termination_manager)
        self.reward_manager = RewardManager(self.cfg.reward, self.sim_data, self.robot_data)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # self.command_manager = CommandManager(self.cfg.command, self.sim_data, self.robot_data)
        # print("[INFO] command Manager: ", self.command_manager)
        self.obs_manager = ObservationManager(self.cfg.obs, self.sim_data, self.robot_data)
        print("[INFO] observation Manager: ", self.obs_manager)
        self.action_manager = ActionManager(self.cfg.action, self.sim_data, self.robot_data)
        print("[INFO] action Manager: ", self.action_manager)
        self.event_manager = EventManager(self.cfg.event, self.sim_data, self.robot_data)
        print("[INFO] event Manager: ", self.event_manager)
        # self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self.sim_data, self.robot_data)
        # print("[INFO] command Manager: ", self.curriculum_manager)

        # apply startup term in event manager
        self.event_manager.apply_startup_terms()

    #--------------- step func ---------------
    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self.robot_data.action = self.action_manager.process_action(action.to(self.sim_data.device))
        # step physics and render each frame
        self.render_gui()

        # perform physics stepping
        for _ in range(self.sim_data.decimation):
            self.robot_data.torque = self.action_manager.compute_torque(self.robot_data.action).float()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_data.torque))
            self.gym.simulate(self.sim)
            if self.sim_data.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # -- post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # -- update env counters
        self.robot_data.episode_length_buf += 1  # step in current episode (per env)
        self.robot_data.common_step_counter += 1  # total step (common for all envs)

        # prepare quantities
        self.robot_data.base_quat[:] = self.robot_data.root_state[:, 3:7]
        self.robot_data.base_lin_vel[:] = quat_rotate_inverse(self.robot_data.base_quat, self.robot_data.root_state[:, 7:10])
        self.robot_data.base_ang_vel[:] = quat_rotate_inverse(self.robot_data.base_quat, self.robot_data.root_state[:, 10:13])
        self.robot_data.projected_gravity[:] = quat_rotate_inverse(self.robot_data.base_quat, self.robot_data.gravity_vec)

        """
        Post physics step callback
        """

        # -- check terminations
        self.robot_data.reset_buf = self.termination_manager.compute()
        self.robot_data.reset_terminated = self.termination_manager.terminated
        self.robot_data.reset_time_out = self.termination_manager.time_out

        # -- reward computation
        self.reward_buf = self.reward_manager.compute()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.robot_data.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # -- update command
        # self.command_manager.compute()

        # -- step interval&tirgger events
        self.event_manager.apply()

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.robot_data.obs = self.obs_manager.compute()

        # self._draw_debug_vis()
        return self.robot_data.obs, self.reward_buf, self.robot_data.reset_buf, self.robot_data.extras

    def reset(self, env_ids):
        # self.curriculum_manager.compute(env_idx = env_idx)
        self.robot_data.extras["log"] = dict()
        info = self.reward_manager.reset(env_ids)
        self.robot_data.extras["log"].update(info)
        info = self.action_manager.reset(env_ids)
        self.robot_data.extras["log"].update(info)
        info = self.termination_manager.reset(env_ids)
        self.robot_data.extras["log"].update(info)
        info = self.event_manager.reset(env_ids)
        self.robot_data.extras["log"].update(info)
        info = self.obs_manager.reset(env_ids)
        self.robot_data.extras["log"].update(info)
        self._reset_robots(env_ids)

    def _reset_robots(self, env_ids):
        """reset func, dof_state/root_state is reseted by event manager"""
        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



    # ---------- creater function ----------
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.sim_data.terrain.static_friction
        plane_params.dynamic_friction = self.sim_data.terrain.dynamic_friction
        plane_params.restitution = self.sim_data.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        # self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
        #                                                                     self.terrain.tot_cols).to(self.device)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.sim_data.terrain.static_friction
        hf_params.dynamic_friction = self.sim_data.terrain.dynamic_friction
        hf_params.restitution = self.sim_data.terrain.restitution

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.sim_data.terrain.static_friction
        tm_params.dynamic_friction = self.sim_data.terrain.dynamic_friction
        tm_params.restitution = self.sim_data.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. set origin of envs,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.sim_data.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = self.sim_data.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.sim_data.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.sim_data.asset.flip_visual_attachments
        asset_options.fix_base_link = self.sim_data.asset.fix_base_link
        asset_options.density = self.sim_data.asset.density
        asset_options.angular_damping = self.sim_data.asset.angular_damping
        asset_options.linear_damping = self.sim_data.asset.linear_damping
        asset_options.max_angular_velocity = self.sim_data.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.sim_data.asset.max_linear_velocity
        asset_options.armature = self.sim_data.asset.armature
        asset_options.thickness = self.sim_data.asset.thickness
        asset_options.disable_gravity = self.sim_data.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        self.sim_data.num_dofs = self.num_dofs
        self.sim_data.num_bodies = self.num_bodies
        if self.num_dofs != self.sim_data.num_actions:
            print(f"[WARNING]Action num is {self.sim_data.num_actions} != Dof num is {self.num_dof}.")

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        print("Check Body Names: ", body_names)

        base_init_state_list = self.sim_data.init_state.pos + self.sim_data.init_state.rot + self.sim_data.init_state.lin_vel + self.sim_data.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.sim_data.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(self.sim_data.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.sim_data.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.sim_data.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "robot", i,
                                                  self.sim_data.asset.self_collisions, 0)
            # dof_props = self._process_dof_props(dof_props_asset, i)
            # self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)

        # # if recording video, set up camera
        # if self.sim_data.env.record_video:
        #     self.camera_props = gymapi.CameraProperties()
        #     self.camera_props.width = 360
        #     self.camera_props.height = 240
        #     self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
        #     self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
        #                                  gymapi.Vec3(0, 0, 0))
        #     if self.eval_cfg is not None:
        #         self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
        #                                                                    self.camera_props)
        #         self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
        #                                      gymapi.Vec3(1.5, 1, 3.0),
        #                                      gymapi.Vec3(0, 0, 0))
        # self.video_writer = None
        # self.video_frames = []
        # self.video_frames_eval = []
        # self.complete_video_frames = []
        # self.complete_video_frames_eval = []

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.sim_data.terrain.mesh_type in ["heightfield", "trimesh"]:
            # self.custom_origins = True
            # self.env_origins = torch.zeros(self.sim_data.num_envs, 3, device=self.sim_data.device, requires_grad=False)
            # # put robots at the origins defined by the terrain
            # max_init_level = self.sim_data.terrain.max_init_terrain_level
            # if not self.sim_data.terrain.curriculum: max_init_level = self.sim_data.terrain.num_rows - 1
            # self.terrain_levels = torch.randint(0, max_init_level+1, (self.sim_data.num_envs,), device=self.sim_data.device)
            # self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            # self.max_terrain_level = self.cfg.terrain.num_rows
            # self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            pass
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.sim_data.num_envs, 3, device=self.sim_data.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.sim_data.num_envs))
            num_rows = np.ceil(self.sim_data.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.sim_data.terrain.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.sim_data.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.sim_data.num_envs]
            self.env_origins[:, 2] = 0.

    # ---------- viewer function ----------
    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                if evt.action == 'w' and evt.value > 0:
                    self.camera_pos += 2.0 * self.camera_direction
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 's' and evt.value > 0:
                    self.camera_pos -= 2.0 *  self.camera_direction
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'e' and evt.value > 0:
                    self.theta -= 0.1
                    self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])
                    self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'q' and evt.value > 0:
                    self.theta += 0.1
                    self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])
                    self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'd' and evt.value > 0:
                    self.camera_pos -= 2.0 * self.camera_direction2
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'a' and evt.value > 0:
                    self.camera_pos += 2.0 *  self.camera_direction2
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)

            # fetch results
            if self.sim_data.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def prepare_viewer_keyboard(self):
        if self.sim_data.headless == False:
            print("Set Headless")
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "w")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "a")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "s")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "d")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "e")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "q")
            self.theta = 0.0
            self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])
            self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
            self.camera_pos = np.array(self.sim_data.viewer.pos, dtype=float)
            self.camera_lookat = np.array(self.sim_data.viewer.lookat, dtype=float)

    def _set_camera(self,pos,lookat):
        cam_pos = gymapi.Vec3(*pos)
        cam_target = gymapi.Vec3(*lookat)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
