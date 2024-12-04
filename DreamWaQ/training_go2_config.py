# !!!Only for Go2
from legged_gym.envs.Go2.go2_basic_config import Go2basicCfg,BasicRunnerCfg
import numpy as np 


class go2EnvCfg(Go2basicCfg):
    class env(Go2basicCfg.env):
        num_envs = 4096
        num_observation_history = 20

        # ----------- Basic Observation ------------
        observe_vel = False
        observe_only_ang_vel = True
        observe_contact_states = False 
        observe_command = True 

        ## 手动 gait 设定的 task, 必须 observe gait command, clock 和 giat indice 才有意义
        observe_two_prev_actions = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_imu = False

        # ---------- Privileged Observations ----------
        num_privileged_obs = 17*11 + 3 + 3 + 4 * 3 * 3 + 4 * 2#17 11 measure height and measure foot height is in terrain
        # privileged_future_horizon = 1
        priv_observe_body_velocity = True
        priv_observe_friction = False #! 1
        priv_observe_restitution = False #! 1 
        priv_observe_base_mass = False #! 1
        priv_observe_com_displacement = True #! 3
        priv_observe_motor_strength = False #! 12
        priv_observe_force_apply = False
        priv_observe_torque_mask = False
        priv_observe_contact_states = True # 4 * 2

        # ---------- Utils ----------
        need_other_obs_state = True

    class terrain(Go2basicCfg.terrain):   
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        
        min_init_terrain_level = 0
        max_init_terrain_level = 0

        # rough terrain only:
        measure_heights = True

        curriculum = True # True
        selected = False # False # select a unique terrain type and pass all arguments
        
        measure_foot_heights = True #! 3 * 3 * 4 

        estimation_height = True # CENet 估计地面高度，现阶段跑下来效果一般，提升基本不大(正常地形一样走，台阶一样难上). 
        estimation_height_points_x = np.linspace(-0.1,0.1,2).tolist()
        estimation_height_points_y = np.linspace(-0.3,0.3,7).tolist()

    class commands(Go2basicCfg.commands):
        command_curriculum = False  # command curriculum现在非常简陋，谨慎和台阶一起开启
        
        num_commands = 3
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y =  [-0.6, 0.6]  # min max [m/s]
        ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s]

    class rewards(Go2basicCfg.rewards):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.3
        reset_force = 400. # 调大防止自杀，同时上台阶时轻微的头部碰撞去感知地形是允许的
        reset_stumble = False # 在一定时间以后，如果累计速度误差（奖励误差的方式计算）达到一定值，就reset
        stumble_vel_error = 0.4
        stumble_second = 5
        
    class reward_scales:
        termination = -5.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5

        lin_vel_z = -2.0
        ang_vel_xy = -0.05 
        orientation = -0.2
        # orientation = -1
        dof_acc = -2.5e-7
        joint_power = -2.e-5
        base_height = -1.0
        # feet_clearance = -0.01
        action_rate = -0.01
        collision = -0.1
        
        # action_smoothness_1 = -0.01
        action_smoothness_2 = -0.01

    class domain_rand(Go2basicCfg.domain_rand):
        rand_interval_s = 15
        randomize_rigids_after_start = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # increase range
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1.0, 2.0]
        randomize_com_displacement = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.05, 0.05]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_lag_timesteps = True
        lag_timesteps = 6
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]

class RunnerCfg(BasicRunnerCfg):
    class algorithm(BasicRunnerCfg.algorithm):
        vae_learning_rate = 5.e-4
        kl_weight = 1.
        
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        encoder_hidden_dims = [256,128,64]
        decoder_hidden_dims = [64,128,256]

        num_history = 20
        num_latent = 16 
        num_estimation = 3 + 14 # vel & height
        num_height_estimation = 14
        activation = 'elu'
    class runner:
        run_name = 'Test'
        experiment_name = 'DreamWaQ_V2.0'
        
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000 # number of policy updates
        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt 

# NOT APPLIED YET
class PlayEnvCfg(go2EnvCfg):
    class env(go2EnvCfg.env):
        num_envs = 50
    
    class terrain(go2EnvCfg.terrain):   
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        terrain_kwargs = {'type':"random_uniform_terrain",
                          'min_height':-0.05,
                          'max_height':0.05,
                          'step':0.005} # None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 2 # number of terrain rows (levels)
        num_cols = 2 # number of terrain cols (types)

    class domain_rand(go2EnvCfg.domain_rand):
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_rigids_after_start = False
        randomize_friction = True
        randomize_restitution = False
        randomize_base_mass = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_motor_strength = True
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        randomize_gravity = False
        push_robots = True
        randomize_lag_timesteps = True
    