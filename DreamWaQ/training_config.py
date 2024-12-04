# This config is used for train.
# It will be the subclass in learning module config.
import numpy as np

class BasicCfg:
    class env:
        num_envs = 4096
        num_observations = 45
        num_actions = 12
        num_observation_history = 20
        episode_length_s = 20  # episode length in seconds

        # ----------- Basic Observation ------------
        ## gravity + dof_pos + dof_vel + action= 3 + 12 + 12 + 12 = 39
        ##
        observe_command = True #! 15
        observe_two_prev_actions = False #! 24
        observe_timing_parameter = False
        observe_clock_inputs = False #! 4
        observe_vel = False #! 6
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_foot_in_base = False

        observe_height_command = False
        observe_gait_commands = False

        observe_imu = True

        # ---------- Privileged Observations ----------
        num_privileged_obs = 2
        # privileged_future_horizon = 1
        priv_observe_friction = True
        priv_observe_friction_indep = False # not implemented
        priv_observe_ground_friction = False # not implemented
        priv_observe_ground_friction_per_foot = False # not implemented
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = False
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

        # ----------- Vedio Recording ------------
        record_video = False
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = True # True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # Height Map only:
        measure_heights = True
        measured_points_x = np.linspace(-0.8,0.8,17).tolist()
        measured_points_y = np.linspace(-0.5,0.5,11).tolist()
        # Footclearance only:
        measure_foot_clearance = False #! 4
        # Footheight only:
        foot_offset = 0.02
        measure_foot_heights = True #! 3 * 3 * 4
        measured_foot_points_x = [-0.1,0.0,0.1]
        measured_foot_points_y = [-0.1,0.0,0.1]

        selected = False # False # select a unique terrain type and pass all arguments
        terrain_kwargs = {
            "plane_terrain":{
                "weight": 1.0,
                "height" : 0.0
            },
            'random_uniform_terrain': {
                "weight": 2.0,
                "min_height" : -0.12,
                "max_height" : 0.12,
                "step" : 0.01,
                "downsampled_scale" : 0.15
            },
            # 'sloped_terrain': {
            #     "weight": 1.0,
            #     "slope" : 0.6
            # },
            'pyramid_sloped_terrain': {
                "weight": 2.0,
                "slope" : -0.7,
                'platform_size': 1.5
            },
            # 'wave_terrain':{
            #     "weight": 2.0,
            #     'num_waves': 2,
            #     'amplitude': 0.7,
            # },
            # 'stairs_terrain':{
            #     "weight": 1.0,
            #     'step_height': 0.15,
            #     'step_width': 0.5,
            # },
            'pyramid_stairs_terrain':{
                "weight": 4.0,
                'step_width': 0.6,
                'step_height': -0.10,
                'platform_size': 2.5,
            },
            # 'stepping_stones_terrain':{
            #     "weight": 1.0,
            #     'stone_size': 1.5,
            #     'stone_distance': 0.1,
            #     'max_height': 0.0,
            #     'platform_size': 4.,
            #     'depth': -10
            # },
            'discrete_obstacles_terrain':{
                "weight": 1.0,
                "max_height":0.2,
                "min_size" :1.0,
                "max_size" : 2.0,
                "num_rects" : 20,
                "platform_size":1.
            },

        }
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 5 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        min_init_terrain_level = 0
        max_init_terrain_level = 0
        center_robots = False
        center_span = 4
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        x_offset = 0.
        teleport_robots = False

    class commands:
        command_curriculum = False

        cmd_cfg = {
            0:{
                'name':'vel_x',
                'init_low':-1.0,
                'init_high':1.0,
                'limit_low':-5.0,
                'limit_high':5.0,
                'local_range':0.1,
                'num_bins':21,
            },
            1:{
                'name':'vel_y',
                'init_low':-0.6,
                'init_high':0.6,
                'limit_low':-0.6,
                'limit_high':0.6,
                'local_range':0.1,
                'num_bins':21,
            },
            2:{
                'name':'vel_yaw',
                'init_low':-1.0,
                'init_high':1.0,
                'limit_low':-5.0,
                'limit_high':5.0,
                'local_range':0.1,
                'num_bins':21,
            }

        }


        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 15
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = True
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        #! 这是 初始的 low 和 high
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y =  [-0.6, 0.6]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd =  [-0.25, 0.15]
        gait_phase_cmd_range =  [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range =  [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        body_roll_range = [-0.0, 0.0]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]

        impulse_height_commands = False

        #! 这是 curriculum 的 low 和 high, 也是能够拓展到的最大范围
        limit_vel_x = [-1.5, 2.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15]
        limit_gait_phase = [0, 1.0]
        limit_gait_offset = [0, 1.0]
        limit_gait_bound = [0, 1.0]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height =  [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll =  [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]



        exclusive_phase_offset = False
        binary_phases = True
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class curriculum_thresholds:
        tracking_lin_vel = 0.7  # closer to 1 is tighter
        tracking_ang_vel = 0.6
        tracking_contacts_shaped_force = 0.9 # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.9

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

    class control:
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        control_type = 'P' #'P'  # P: position, V: velocity, T: torques
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_v2.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on =["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        fix_base_link = False
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True #!
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        # add link masses, increase range, randomize inertia, randomize joint properties
        rand_interval_s = 15
        randomize_rigids_after_start = False
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.05, 0.05]
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]

        gravity_rand_interval_s = 8
        gravity_impulse_duration = 0.99
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]

        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6

    class rewards:
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 0.02

        # reward_container_name = "CoRLRewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.3
        max_contact_force = 100.  # forces above this value are penalized

        #! not immplemented in complex terrain
        use_terminal_body_height = True
        terminal_body_height = 0.05
        terminal_foot_height = -0.005
        terminal_body_ori = 1.6

        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        footswing_height = 0.09

        reset_force = 100.
        reset_stumble = False
        stumble_vel_error = 0.8
        stumble_second = 5

    class reward_scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.2
        # orientation_control = -5.0
        # dof_vel = -1e-4
        dof_acc = -2.5e-7
        body_height = -1.0
        collision = -1.
        action_rate = -0.01
        # tracking_contacts_shaped_force = 4.0
        # tracking_contacts_shaped_vel = 4.0
        # jump = 10.0
        # torques = -0.0001
        # feet_slip = -0.04

        # action_smoothness_1 = -0.01
        action_smoothness_2 = -0.01
        # raibert_heuristic = -10.0
        # feet_clearance_cmd_linear = -30


    class normalization:
        clip_observations = 100.
        clip_actions = 10.0

        friction_range =  [0, 1]
        ground_friction_range =  [0, 1]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        foot_in_base = 2.0
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise:
        add_noise = True
        noise_level = 0.1  # scales other values

    class noise_scales:
        dof_pos = 0.01
        dof_vel = 0.02
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        foot_in_base = 0.01
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [2.0, 2.0, 2.0]  # [m]
        lookat = [1., 1., 0.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class BasicRunnerCfg:

    class algorithm:
        # algorithmpass
        value_loss_coef = 1.0
        contrastive_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.02
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 # 5.e-4
        adaptation_module_learning_rate = 5.e-4
        num_adaptation_module_substeps = 5
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.02
        max_grad_norm = 1.
        selective_adaptation_module_loss = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # adaptation_module_branch_hidden_dims = [256, 128]
        activation = 'elu'

    class runner:
        run_name = 'Test'
        experiment_name = 'GO_Expert'

        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 500 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt


class EnvCfg(BasicCfg):
    class env(BasicCfg.env):
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
        num_privileged_obs = 3#17 11 measure height and measure foot height is in terrain
        # privileged_future_horizon = 1
        priv_observe_body_velocity = True
        priv_observe_friction = False #! 1
        priv_observe_restitution = False #! 1
        priv_observe_base_mass = False #! 1
        priv_observe_com_displacement = False #! 3
        priv_observe_motor_strength = False #! 12
        priv_observe_force_apply = False
        priv_observe_torque_mask = False
        priv_observe_contact_states = False # 4 * 2

        # ---------- Utils ----------
        need_other_obs_state = True

    class terrain(BasicCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]

        min_init_terrain_level = 0
        max_init_terrain_level = 0

        # rough terrain only:
        measure_heights = False

        curriculum = True # True
        selected = False # False # select a unique terrain type and pass all arguments

        measure_foot_heights = False #! 3 * 3 * 4

        estimation_height = False # CENet 估计地面高度，现阶段跑下来效果一般，提升基本不大(正常地形一样走，台阶一样难上).
        estimation_height_points_x = np.linspace(-0.1,0.1,2).tolist()
        estimation_height_points_y = np.linspace(-0.3,0.3,7).tolist()

    class commands(BasicCfg.commands):
        command_curriculum = False  # command curriculum现在非常简陋，谨慎和台阶一起开启

        num_commands = 3
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y =  [-0.6, 0.6]  # min max [m/s]
        ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s]

    class rewards(BasicCfg.rewards):
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

    class domain_rand(BasicCfg.domain_rand):
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

        num_history = 6
        num_latent = 16
        num_estimation = 3 # vel & height
        num_height_estimation = 0
        activation = 'elu'
    class runner:
        run_name = 'Test'
        experiment_name = 'DreamWaQ_V2.0'

        num_steps_per_env = 45 # per iteration
        max_iterations = 10000 # number of policy updates
        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

# NOT APPLIED YET
class PlayEnvCfg(EnvCfg):
    class env(EnvCfg.env):
        num_envs = 50

    class terrain(EnvCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        terrain_kwargs = {'type':"random_uniform_terrain",
                          'min_height':-0.05,
                          'max_height':0.05,
                          'step':0.005} # None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 2 # number of terrain rows (levels)
        num_cols = 2 # number of terrain cols (types)

    class domain_rand(EnvCfg.domain_rand):
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
