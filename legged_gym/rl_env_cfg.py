from dataclasses import MISSING
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import *

class ManagerBasedRLEnvCfg:
    sim_data: SimData = None
    # robot_data: RobotData = None
    reward = None
    termination = None
    command = None
    event = None
    obs = None
    action = None
    curriculum = None

# example
class Reward:
    tracking_linear_vel = RewardTerm()
    tracking_angular_vel = RewardTerm()
class Termination:
    time_out = TerminationTerm()
# ......
class cfg(ManagerBasedRLEnvCfg):
    sim_data = SimData()
    sim_data.num_envs = 1

    reward = Reward()
    termination = Termination()