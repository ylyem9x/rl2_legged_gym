import os
from datetime import datetime
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from DreamWaQ.training_config import EnvCfg,RunnerCfg
from DreamWaQ.runners.on_policy_runner import OnPolicyRunner
from DreamWaQ.modules.actor_critic import ActorCritic
from legged_gym.rl_env import ManagerBasedRLEnv
import torch

def launch(args, path, env_cfg, train_cfg):

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    from legged_gym.envs.base.base import cfg
    env = ManagerBasedRLEnv(cfg)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    # log_root = os.path.join(LEGGED_GYM_ROOT_DIR,'..', 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    _,train_cfg = update_cfg_from_args(None,train_cfg,args)
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env,train_cfg,log_dir,device=args.rl_device)
    if train_cfg.runner.resume == True and path != None:
        runner.load(path)
    return env, runner ,env_cfg ,train_cfg

def play(arg, path, env_cfg, train_cfg):

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 25)

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    # load policy
    headless = False

    device = args.rl_device
    if args.task == 'go1' or args.task == 'go2':
        env = LeggedRobot(sim_params=sim_params,
                                        physics_engine=args.physics_engine,
                                        sim_device=args.sim_device,
                                        headless=args.headless,
                                        cfg = env_cfg)
    else:
        env = HexMini(sim_params=sim_params,
                                        physics_engine=args.physics_engine,
                                        sim_device=args.sim_device,
                                        headless=args.headless,
                                        cfg = env_cfg)
    env = HistoryWrapper(env)
    policy_cfg  = train_cfg.policy
    policy = ActorCritic(
            env.num_obs,
            env.num_privileged_obs,
            env.num_actions,
            policy_cfg.num_latent,
            policy_cfg.num_history,
            policy_cfg.num_estimation,
            policy_cfg.activation,
            policy_cfg.actor_hidden_dims,
            policy_cfg.critic_hidden_dims,
            policy_cfg.encoder_hidden_dims,
            policy_cfg.decoder_hidden_dims,
        ).to(device)

    # env.set_apply_force(0, 50, z_force_norm = 0)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'])
    play_policy(env_cfg,train_cfg,policy,env,cmd_vel = [0.8,0.0,0.0],
                move_camera=False,record=True)

if __name__ == '__main__':
    args = get_args()
    path = "logs/DreamWaQ_V2.0/Oct23_08-17-46_Test/model_10000.pt"

    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    env, runner , env_cfg ,train_cfg = launch(args, path, env_cfg, train_cfg)
    runner.learn(num_learning_iterations=1000)
