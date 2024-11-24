import numpy as np
# ---------- Reward Manager Term ----------
class RewardTerm:
    def __init__(self, func, weight, cfg: dict = None):
        """
        Args:
            func: function for computing reward. input: (sim_data, robot_data, args). return torch.Tensor
            weight: the weight of this reward term. set 0 will disable this reward.
            cfg: cfg that send to the reward function. for example, {"sigma": 0.25, "threshold": 0.5}
        """
        self.func = func
        self.weight = weight
        self.cfg = cfg

class RewardResetFunc:
    """
    overwrite the reset function in reward_manager.
    No prevention here, bugs may occur.
    Not suggested.
    You can see the necessary function's input and output in reward_manager.py
    """
    def __init__(self, func):
        self.func = func

class RewardComputeFunc:
    """
    overwrite the compute function in reward_manager.
    No prevention here, bugs may occur.
    Not suggested.
    You can see the necessary function's input and output in reward_manager.py
    """
    def __init__(self, func):
        self.func = func

# ---------- Termination Manager Term ----------
class TerminationTerm:
    def __init__(self, func, time_out = False, cfg: dict = None):
        """
        Args:
            func: function for computing termination. input: (sim_data, robot_data, args). return torch.Tensor
            time_out: This signal is set to true if the environment has ended naturally.
            cfg: cfg that send to the reward function. for example, {"sigma": 0.25, "threshold": 0.5}
        """
        self.func = func
        self.time_out = time_out
        self.cfg = cfg

# ---------- Action Manager Term ----------
class ActionSacleTerm:
    def __init__(self, scale, clip):
        self.scale = scale
        self.clip = clip

class ActionComputeTerm:
    def __init__(self, control_type, kp = None, kd = None, func = None):
        """
        control_type: "P", "T", "other"

        if "P", kp/kd is needed

        if "other", func is needed. Func's args: (sim_data, robot_data, action_scaled). return torques: torch.Tensor
        """
        self.control_type = control_type
        self.kp = kp
        self.kd = kd
        self.func = func

# ---------- Event Manager Term ----------
class EventTerm:
    def __init__(self, func, cfg = None):
        self.func = func
        self.cfg = cfg

# ---------- Curriculum Manager Term ----------
class CurriculumTerm:
    def __init__(self, func, cfg = None):
        self.func = func
        self.cfg = cfg

# ---------- RobotData Term -----------
class RobotDataTerm:
    def __init__(self, init_func, reset_func, compute_func):
        self.init_func = init_func
        self.reset_func = reset_func
        self.compute_func = compute_func
        self.data = None
    def __call__(self):
        return self.data
