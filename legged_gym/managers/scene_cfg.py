class SimCfg:
    class GymParams:
        physics_engine = "PhysX"
        sim_device = "cuda:0"
        use_gpu_pipeline = False
        headless = False
        dt = 0.005

    class EnvParams:
        num_envs = 4096
        num_actions = 0