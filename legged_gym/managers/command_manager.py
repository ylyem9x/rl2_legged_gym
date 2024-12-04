import torch
import numpy as np
from typing import Dict
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import CommandTerm

class CommandManager:
    """Manager for generating commands.

    The command manager is used to generate commands for an agent to execute. It makes it convenient to switch
    between different command generation strategies within the same environment. For instance, in an environment
    consisting of a quadrupedal robot, the command to it could be a velocity command or position command.
    By keeping the command generation logic separate from the environment, it is easy to switch between different
    command generation strategies.

    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
        """
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self.terms = []
        self.dim = 0
        self._prepare_terms()
        robot_data.command = torch.zeros(sim_data.num_envs, self.dim,
                                         dtype=torch.float, device=sim_data.device, requires_grad=False)

    def __str__(self) -> str:
        """Returns: A string representation for command term."""
        msg = f"<CommandManager> contains {self.num_terms} command term.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Command Terms"
        table.field_names = ["func_name", "reset_name", "params"]
        # add info on each term
        term: CommandTerm
        for term in self.terms:
            table.add_row([term.func.__name__, term.reset.__name__, term.cfg])
        # convert table to string
        msg += table.get_string()
        msg += "\n"
        return msg

    def apply(self):
        term: CommandTerm
        dim = 0
        for term in self.terms:
            this_cmd = self.robot_data.command[:, dim: term.dim + dim].clone()
            self.robot_data.command[:, dim: term.dim + dim] = term.func(self.sim_data, self.robot_data,
                                                                        this_cmd, term.cfg).clone()
            dim += term.dim


    def reset(self, env_ids) -> Dict[str, torch.Tensor]:
        term: CommandTerm
        extras = dict()
        for term in self.terms:
            info = term.reset(self.sim_data, self.robot_data, env_ids, term.cfg)
            extras.update(info)
        return extras

    """
    Helper functions.
    """

    def _prepare_terms(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), CommandTerm)
                   and not name.startswith("__")]
        self.num_terms = len(members)
        term: CommandTerm
        for name in members:
            term = getattr(self.cfg, name)
            if term.cfg == None:
                term.cfg = dict()
            self.terms.append(term)
            self.dim += term.dim
