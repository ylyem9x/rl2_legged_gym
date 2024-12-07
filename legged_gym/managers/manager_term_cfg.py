import numpy as np
# ---------- Reward Manager Term ----------
class RewardTerm:
    def __init__(self, func, weight, cfg = dict()):
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
    def __init__(self, func, time_out = False, cfg = dict()):
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
    def __init__(self, scale, clip = 100.0):
        self.scale = scale
        self.clip = clip

class ActionComputeTerm:
    def __init__(self, control_type, kp = None, kd = None, lag = 1, func = None):
        """
        Args:
            control_type: "P", "T", "other"
            kp,kd: needed when "P"
            lag: length of lag buffer, lagtime = length * dt / decimation
            func: needed when "other". Func's args: (sim_data, robot_data, action_scaled). return torques: torch.Tensor
        """
        self.control_type = control_type
        self.kp = kp
        self.kd = kd
        self.lag = lag
        self.func = func

# ---------- Observation Term ----------
class ObservationTerm:
    def __init__(self, func, noise = None, clip = 100.0, scale = 1.0, cfg = dict()):
        self.func = func
        self.scale = scale
        self.noise = noise # call to noise obs
        self.clip = clip
        self.cfg = cfg

class ObservationGroup:
    def __init__(self, *args, concatenate_terms: bool = True):
        self.concatenate_terms = concatenate_terms
        term: ObservationTerm
        for term in args:
            setattr(self, term.func.__name__, term)

# ---------- Event Manager Term ----------
class EventTerm:
    def __init__(self, func, mode, interval = {"global":15}, cfg = dict()):
        self.func = func
        self.mode = mode # "startup", "reset", "interval", "decimation"
        self.interval = interval # "global", "local"
        self.cfg = cfg # for interval: global for all env, local will input env_ids

# ---------- Command Manager Term -----------
class CommandTerm:
    def __init__(self, dim, func, reset, cfg = dict()):
        self.dim = dim
        self.func = func # every dt, keep or resample
        self.reset = reset # reset extras, can apply curriculum
        self.cfg = cfg

# ---------- Robot Data Term -----------
class RobotDataTerm:
    def __init__(self, name, init, compute=None, reset=None):
        self.name = name
        self.init = init
        self.compute = compute # only data from sim is refreshed
        self.reset = reset