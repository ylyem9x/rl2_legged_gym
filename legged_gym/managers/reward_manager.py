import torch
import numpy as np
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData

from legged_gym.managers.manager_term_cfg import RewardTerm, RewardResetFunc, RewardComputeFunc

class RewardManager:
    """Manager for computing reward signals from SimData and RobotData.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from array of RewardTerm.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary.
            env: The environment instance.
        """
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self.resetfunc = None
        self.computefunc = None
        self._episode_sums = dict()
        self._prepare_terms()
        for func,_,_ in self.terms:
            self._episode_sums[func.__name__] = torch.zeros(sim_data.num_envs,
                                                            dtype=torch.float, device=sim_data.device)
        self._reward_buf = torch.zeros(sim_data.num_envs, dtype=torch.float, device=sim_data.device)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self.num_func)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Name", "Weight", "Cfg"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for func,weight,cfg in self.terms:
            table.add_row([func.__name__, weight, cfg])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    def compute(self) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        if self.computefunc == None:
            # reset computation
            self._reward_buf[:] = 0.0
            # iterate over all the reward terms
            for func,weight,args in self.terms:
                # skip if weight is zero (kind of a micro-optimization)
                if weight == 0.0:
                    continue
                # compute term's value
                value = func(self.sim_data, self.robot_data, args) * weight * self.sim_data.dt
                # update total reward
                self._reward_buf += value
                # update episodic sum
                self._episode_sums[func.__name__] += value
        else:
            self._reward_buf, self._episode_sums = self.computefunc(self.sim_data, self.robot_data, self.terms)

        return self._reward_buf

    def reset(self, env_ids) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        if self.resetfunc == None:
            # resolve environment ids
            if env_ids is None:
                env_ids = slice(None)
            # store information
            extras = {}
            for key in self._episode_sums.keys():
                # store information
                # r_1 + r_2 + ... + r_n
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.sim_data.max_episode_length_s
                # reset episodic sum
                self._episode_sums[key][env_ids] = 0.0
        else:
            extras = self.resetfunc(self.sim_data, self.robot_data, env_ids, self._episode_sums, self.terms)
        # return logged information
        return extras

    def reprepare_terms(self, cfg, set_zero = False):
        """used to update from new cfg
        Args:
            set_zero: set self._episode_sums and self._rew_buf to zero
        """
        self.cfg = cfg
        self.resetfunc = None
        self.computefunc = None
        self._prepare_terms()
        if set_zero:
            self._episode_sums = dict()
            for func,_,_ in self.terms:
                self._episode_sums[func.__name__] = torch.zeros(self.sim_data.num_envs,
                                                                dtype=torch.float, device=self.sim_data.device)
            self._reward_buf = torch.zeros(self.sim_data.num_envs, dtype=torch.float, device=self.sim_data.device)

    """
    Helper functions.
    """

    def _prepare_terms(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), RewardTerm)
                   and not name.startswith("__")]
        self.num_func = len(members)
        if self.num_func == 0:
            raise ValueError(f"No term of type RewardTerm")

        self.terms = []
        term: RewardTerm
        for name in members:
            term = getattr(self.cfg, name)
            func = term.func
            w = term.weight
            cfg = term.cfg
            if not (isinstance(w, int) or isinstance(w, float)):
                raise TypeError(
                    f"Weight of '{name}' is not int or float"
                )
            self.terms.append((func, w, cfg))

        # check the function overwrite
        Resetmembers = [name for name in attributes if
                            isinstance(getattr(self.cfg, name), RewardResetFunc)
                            and not name.startswith("__")]
        if len(Resetmembers) == 1:
            self.resetfunc = getattr(self.cfg, Resetmembers[0])
        elif len(Resetmembers) > 1:
            print("[WARNING] RewardReset Functions conflict! No overwriting will be applied.")

        Computemembers = [name for name in attributes if
                            isinstance(getattr(self.cfg, name), RewardComputeFunc)
                            and not name.startswith("__")]
        if len(Computemembers) == 1:
            self.computefunc = getattr(self.cfg, Computemembers[0])
        elif len(Computemembers) > 1:
            print("[WARNING] RewardCompute Functions conflict! No overwriting will be applied.")