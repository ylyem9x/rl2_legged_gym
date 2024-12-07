import torch
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.data import SimData, RobotData
from legged_gym.utils.math import torch_rand

class basic_event:
    # ---------- interval func ----------
    def random_changing_base_vel(sim_data:SimData, robot_data:RobotData, cfg:dict):
        max_vel = cfg.get("max",1.0)
        sim, gym, robot_asset, envs, actor_handles = robot_data.gym_env
        robot_data.root_state[:, 7:9] = torch_rand(-max_vel, max_vel, (sim_data.num_envs, 2), device=sim_data.device) # lin vel x/y
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(robot_data.root_state))

    # ---------- reset func -----------
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
        return {}

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
        return {}

    def terrain_curriculum(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        """
        update terrain curriculum by changing the origin of the robot
        MUST BEFORE ROOT RESET!
        """
        if not hasattr(robot_data, "terrain_levels"):
            robot_data.terrain_origins = torch.from_numpy(robot_data.terrain.env_origins).to(sim_data.device).to(torch.float)
            robot_data.terrain_levels = torch.from_numpy(robot_data.terrain.terrain_levels).to(sim_data.device).to(torch.long)
            robot_data.terrain_types = torch.from_numpy(robot_data.terrain.terrain_types).to(sim_data.device).to(torch.long)
        distance = torch.norm(robot_data.root_state[env_ids, :2] - robot_data.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > robot_data.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(robot_data.command[env_ids, :2], dim=1)*sim_data.max_episode_length_s*0.5) * ~move_up
        robot_data.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        robot_data.terrain_levels[env_ids] = torch.where(robot_data.terrain_levels[env_ids]>=robot_data.terrain.cfg.num_rows - 1,
                                                   torch.randint_like(robot_data.terrain_levels[env_ids], robot_data.terrain.cfg.num_rows - 1),
                                                   torch.clip(robot_data.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        robot_data.env_origins[env_ids] = robot_data.terrain_origins[robot_data.terrain_levels[env_ids], robot_data.terrain_types[env_ids]]
        return {"mean_terrain_level":torch.mean(robot_data.terrain_levels.float()).item()}

    def random_PD_gain(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        kp_factor = cfg.get("kp_factor", [0.9, 1.1])
        kd_factor = cfg.get("kd_factor", [0.9, 1.1])
        p_gain = torch_rand(kp_factor[0], kp_factor[1], (len(env_ids), 1), sim_data.device)
        d_gain = torch_rand(kd_factor[0], kd_factor[1], (len(env_ids), 1), sim_data.device)
        robot_data.p_gain[env_ids] = p_gain
        robot_data.d_gain[env_ids] = d_gain
        return {}

    def random_motor_strength(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        strength = cfg.get("strength", [0.9, 1.1])
        motor_strength = torch_rand(strength[0], strength[1], (len(env_ids), 1), sim_data.device)
        robot_data.motor_strength[env_ids] = motor_strength
        return {}

    def random_motor_offset(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        offset = cfg.get("offset", [-0.1, 0.1])
        motor_offset = torch_rand(offset[0], offset[1], (len(env_ids), 1), sim_data.device)
        robot_data.motor_offset[env_ids] = motor_offset
        return {}

    # ---------- decimation func ----------
    def apply_force(sim_data:SimData, robot_data:RobotData, env_ids, cfg:dict):
        max_force = cfg.get("max",10.)


    # ---------- startup func ----------
    def random_base_props(sim_data:SimData, robot_data:RobotData, cfg:dict):
        """
        Args of cfg:
            "payload": (lower_limit, upper_limit)
            "com_displacement":  (lower_limit, upper_limit)
        """
        sim, gym, robot_asset, envs, actor_handles = robot_data.gym_env
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
        sim, gym, robot_asset, envs, actor_handles = robot_data.gym_env
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