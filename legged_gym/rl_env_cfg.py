from dataclasses import MISSING
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import *

class ManagerBasedRLEnvCfg:
    sim_data = SimData()
    robot_data_terms = []
    print_manager = True
    reward = None
    termination = None
    command = None
    event = None
    obs = None
    action = None
    curriculum = None
    terrain = None