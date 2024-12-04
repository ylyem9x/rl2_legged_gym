import torch
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.data import SimData, RobotData
from legged_gym.utils.math import torch_rand

class basic_event:
    # reset func
    def reset_dof(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        """Reset dof with random init pos from multiply uniform random scale
        Velocities of dofs are set to zero.

        Args of cfg:
            "limit": (lower_limit, upper_limit)
        """
        lower_limit, upper_limit = cfg.get("limit", (0.5, 1.5))
        robot_data.dof_pos[env_ids] = robot_data.default_dof_pos * torch_rand(lower_limit, upper_limit, (len(env_ids), sim_data.num_dofs),
                                                                                    device=sim_data.device)
        robot_data.dof_vel[env_ids] = 0.

    def reset_root_states(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        """
        Args of cfg:
            "random_vel_range": float
        """
        base_init_state = robot_data.extra_gym_info["base_init_state"]
        vel_range = cfg.get("random_vel_range", 0.0)
        if sim_data.terrain.mesh_type != "plane":
            robot_data.root_state[env_ids] = base_init_state
            robot_data.root_state[env_ids, :3] += robot_data.env_origins[env_ids]
            robot_data.root_state[env_ids, :2] += torch_rand(-1., 1., (len(env_ids), 2), device=sim_data.device) # xy position
        else:
            robot_data.root_state[env_ids] = base_init_state
            robot_data.root_state[env_ids, :3] += robot_data.env_origins[env_ids]
        # base velocities
        robot_data.root_state[env_ids, 7:13] = torch_rand(-vel_range, vel_range, (len(env_ids), 6), device=sim_data.device) # [7:10]: lin vel, [10:13]: ang vel

    # startup func
    def random_base_props(sim_data:SimData, robot_data:RobotData, cfg:dict):
        """
        Args of cfg:
            "payload": (lower_limit, upper_limit)
            "com_displacement":  (lower_limit, upper_limit)
        """
        gym, robot_asset, envs, actor_handles = robot_data.gym_env
        payload_lower_limit, payload_uppper_limit = cfg.get("payload",(0.0,0.0))
        com_lower_limit, com_uppper_limit = cfg.get("com_displacement",(0.0,0.0))
        robot_data.payloads = torch_rand(payload_lower_limit, payload_uppper_limit, (sim_data.num_envs, 1), device=sim_data.device).squeeze(-1)
        robot_data.com_displacement = torch_rand(com_lower_limit, com_uppper_limit, (sim_data.num_envs, 3), device=sim_data.device)
        robot_data.default_base_mass = gym.get_actor_rigid_body_properties(envs[0], actor_handles[0])[0].mass
        for i in range(sim_data.num_envs):
            body_props = gym.get_actor_rigid_body_properties(envs[i], actor_handles[i])
            body_props[0].mass += robot_data.payloads[i]
            body_props[0].com = gymapi.Vec3(robot_data.com_displacement[i,0], robot_data.com_displacement[i,1], robot_data.com_displacement[i,2])
            gym.set_actor_rigid_body_properties(envs[i], actor_handles[i], body_props, recomputeInertia=True)

    def random_rigid_props(sim_data:SimData, robot_data:RobotData, cfg:dict):
        """
        Args of cfg:
            "friction": (lower_limit, upper_limit)
            "restitution":  (lower_limit, upper_limit)
        """
        gym, robot_asset, envs, actor_handles = robot_data.gym_env
        fri_lower_limit, fri_uppper_limit = cfg.get("friction",(0.0,0.0))
        res_lower_limit, res_uppper_limit = cfg.get("restitution",(0.0,0.0))
        rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)
        num_rigid = len(rigid_shape_props_asset)
        num_buckets = cfg.get("num_buckets", 64)
        fri_bucketa_ids = torch.randint(0, num_buckets, (sim_data.num_envs, 1))
        resti_bucket_ids = torch.randint(0, num_buckets, (sim_data.num_envs, 1))
        friction_bucket = torch_rand(fri_lower_limit, fri_uppper_limit, (num_buckets, num_rigid), device=sim_data.device)
        resti_bucket = torch_rand(res_lower_limit, res_uppper_limit, (num_buckets, num_rigid), device=sim_data.device)
        robot_data.friction = friction_bucket[fri_bucketa_ids].squeeze(1)
        robot_data.restitution = resti_bucket[resti_bucket_ids].squeeze(1)
        for i in range(sim_data.num_envs):
            for s in range(num_rigid):
                rigid_shape_props_asset[s].friction = robot_data.friction[i, s]
                rigid_shape_props_asset[s].restitution = robot_data.restitution[i, s]
            gym.set_actor_rigid_shape_properties(envs[i], actor_handles[i], rigid_shape_props_asset)