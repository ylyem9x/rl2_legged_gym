import torch
import numpy as np
from typing import Dict
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import EventTerm

class EventManager:
    """Manager for orchestrating operations based on different simulation events.

    The event manager applies operations to the environment based on different simulation events. For example,
    changing the masses of objects or their friction coefficients during initialization/ reset, or applying random
    pushes to the robot at a fixed interval of steps. The user can specify several modes of events to fine-tune the
    behavior based on when to apply the event.

    Event terms can be grouped by their mode. The mode is a user-defined string that specifies when
    the event term should be applied. This provides the user complete control over when event
    terms should be applied.

    For a typical training process, you may want to apply events in the following modes:

    - "startup": Event is applied once at the beginning of the training.
    - "reset": Event is applied at every reset.
    - "interval": Event is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.
    For this you will need to add the triggering of that mode in the environment implementation as well.

    .. note::

        The triggering of operations corresponding to the mode ``"interval"`` are the only mode that are
        directly handled by the manager itself. The other modes are handled by the environment implementation.

    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self.startup_terms = []
        self.reset_terms = []
        self.interval_terms = []
        self.decimation_terms = []
        self._prepare_terms()

    def __str__(self) -> str:
        """Returns: A string representation for event term."""
        msg = f"<EventManager> contains {self.num_terms} event term.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Event Terms"
        table.field_names = ["name", "mode", "params"]
        # add info on each term
        for i in range(len(self.startup_terms)):
            term: EventTerm = self.startup_terms[i]
            table.add_row([term.func.__name__, "startup", term.cfg])
        for i in range(len(self.reset_terms)):
            term: EventTerm = self.reset_terms[i]
            table.add_row([term.func.__name__, "reset", term.cfg])
        for i in range(len(self.interval_terms)):
            term: EventTerm = self.interval_terms[i]
            table.add_row([term.func.__name__, f"inteval:{term.interval}", term.cfg])
        for i in range(len(self.decimation_terms)):
            term: EventTerm = self.decimation_terms[i]
            table.add_row([term.func.__name__, "decimation", term.cfg])
        # convert table to string
        msg += table.get_string()
        msg += "\n"
        return msg

    def apply_decimation_terms(self):
        term: EventTerm
        for term in self.decimation_terms:
            term.func(self.sim_data, self.robot_data, term.cfg)

    def apply(self):
        """apply interval terms"""
        term: EventTerm
        for term in self.interval_terms:
            if "global" in term.interval:
                t = int(term.interval["global"] / self.sim_data.dt)
                if self.robot_data.common_step_counter % t == 0:
                    term.func(self.sim_data, self.robot_data, term.cfg)
            elif "local" in term.interval:
                interval = int(term.interval["local"] / self.sim_data.dt)
                env_ids = (self.robot_data.episode_length_buf % interval == 0).nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    term.func(self.sim_data, self.robot_data, env_ids, term.cfg)

    def apply_startup_terms(self):
        """apply startup terms"""
        for i in range(len(self.startup_terms)):
            term: EventTerm = self.startup_terms[i]
            term.func(self.sim_data, self.robot_data, term.cfg)

    def reset(self, env_ids) -> Dict[str, torch.Tensor]:
        extras = dict()
        for i in range(len(self.reset_terms)):
            term: EventTerm = self.reset_terms[i]
            info = term.func(self.sim_data, self.robot_data, env_ids, term.cfg)
            extras.update(info)
        return extras

    """
    Helper functions.
    """

    def _prepare_terms(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), EventTerm)
                   and not name.startswith("__")]
        self.num_terms = len(members)
        term: EventTerm
        for name in members:
            term = getattr(self.cfg, name)
            func = term.func
            mode = term.mode
            if term.cfg == None:
                term.cfg = dict()
            if mode == "startup":
                self.startup_terms.append(term)
            elif mode == "reset":
                self.reset_terms.append(term)
            elif mode == "interval":
                self.interval_terms.append(term)
            elif mode == "decimation":
                self.decimation_terms.append(term)
            else:
                raise ValueError(f"Wrong mode value, expected startup/reset/interval/decimation, but recevied f{mode}")
