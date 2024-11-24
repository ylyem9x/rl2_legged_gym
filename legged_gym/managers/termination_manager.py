import torch
import numpy as np
from typing import Dict
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import TerminationTerm

class TerminationManager:
    """Manager for computing done signals for a given world.

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which returns a boolean tensor of shape
    (num_envs,). The termination managercomputes the termination signal as the union (logical or)
    of all the termination terms.

    * **Time-out**: This signal is set to true if the environment has ended naturally after an externally defined condition.
      For example, the environment may be terminated if the episode has timed out
      (i.e. reached max episode length).
    * **Terminated**: This signal is set to true if the environment has reached a terminal state defined by the
      environment. This state may correspond to task success, task failure, etc.

    These signals can be individually accessed using the :attr:`time_outs` and :attr:`terminated` properties.

    The termination terms are parsed from array of TerminationTerm.
    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        """Initialize the termination manager.

        Args:
            cfg: The configuration object or dictionary.
        """
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self._term_done = dict()
        self._prepare_terms()
        for func,_,_ in self.terms:
            self._term_done[func.__name__] = torch.zeros(sim_data.num_envs,
                                                        dtype=torch.bool, device=sim_data.device)
        # create buffer for managing termination per environment
        self._truncated_buf = torch.zeros(self.sim_data.num_envs, device=self.sim_data.device, dtype=torch.bool) # time_out
        self._terminated_buf = torch.zeros_like(self._truncated_buf) # Fall

    def __str__(self) -> str:
        """Returns: A string representation for termination manager."""
        msg = f"<TerminationManager> contains {self.num_func} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Termination Terms"
        table.field_names = ["Name", "Time Out", "Cfg"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for func,time_out,cfg in self.terms:
            table.add_row([func.__name__, time_out, cfg])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is (num_envs,)."""
        return self._truncated_buf | self._terminated_buf

    @property
    def time_out(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,).
        """
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,).
        """
        return self._terminated_buf

    def compute(self) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the termination terms
        for func,time_out,args in self.terms:
            reset_buf = func(self.sim_data, self.robot_data, args)
            if time_out:
                self._truncated_buf |= reset_buf
            else:
                self._terminated_buf |= reset_buf
            self._term_done[func.__name__][:] = reset_buf

        return self._truncated_buf | self._terminated_buf

    def reset(self, env_ids) -> Dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._term_done.keys():
            extras["Episode_Termination/" + key] = torch.count_nonzero(self._term_done[key][env_ids]).item()
            # reset episodic sum
            self._term_done[key][env_ids] = False
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
            for func,_,_ in self.terms:
                self._term_done[func.__name__] = torch.zeros(self.sim_data.num_envs,
                                                            dtype=torch.bool, device=self.sim_data.device)
            # create buffer for managing termination per environment
            self._truncated_buf *= 0
            self._terminated_buf *= 0

    """
    Helper functions.
    """

    def _prepare_terms(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), TerminationTerm)
                   and not name.startswith("__")]
        self.num_func = len(members)
        if self.num_func == 0:
            self.terms.append((_reset_func_return_all_false, True, None))
            print("[WARNING]No term of type TerminationTerm. Env will never end.")

        self.terms = []
        term: TerminationTerm
        for name in members:
            term = getattr(self.cfg, name)
            func = term.func
            time_out = term.time_out
            cfg = term.cfg
            self.terms.append((func, time_out, cfg))

def _reset_func_return_all_false(sim_data:SimData, robot_data:RobotData, cfg = None):
    reset_buf = torch.zeros(sim_data.num_envs, device=sim_data.device, dtype=torch.bool)
    return reset_buf
