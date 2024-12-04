import torch
import numpy as np
from typing import Dict
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import ObservationTerm, ObservationGroup

class ObservationManager:
    """Manager for computing observations.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroup` class. Within each group, each
    observation term should instantiate the :class:`ObservationTerm` class. Based on the configuration, the
    observations in a group can be concatenated into a single tensor or returned as a dictionary with keys
    corresponding to the term's name.

    .. note::
        When the observation terms in a group do not have the same shape, the observation terms cannot be
        concatenated. In this case, please set the :attr:`ObservationGroupCfg.concatenate_terms` attribute in the
        group configuration to False.

    The observation manager can be used to compute observations for all the groups or for a specific group. The
    observations are computed by calling the registered functions for each term in the group. The functions are
    called in the order of the terms in the group. The functions are expected to return a tensor with shape
    (num_envs, ...).

    If a noise model or custom modifier is registered for a term, the function is called to corrupt
    the observation. The corruption function is expected to return a tensor with the same shape as the observation.
    The observations are clipped and scaled as per the configuration settings.
    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        """Initialize the termination manager.

        Args:
            cfg: The configuration object or dictionary.
        """
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self._prepare_groups()

    def __str__(self) -> str:
        """Returns: A string representation for Observation manager."""
        msg = f"<TerminationManager> contains {self.num_group} active Groups.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Observation"
        table.field_names = ["Group Name", "Obs Name", "Noise", "Clip", "Scale", "cfg"]
        # add info on each term
        for name, terms, _ in self.groups:
            for term in terms:
                table.add_row([name, term.func.__name__, term.noise, term.clip, term.scale, term.cfg])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    def compute(self) -> torch.Tensor:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
            The observations are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for name, terms, concatenated in self.groups:
            obs_buffer[name] = self.compute_group(terms, concatenated)
        # otherwise return a dict with observations of all groups
        return obs_buffer

    def compute_group(self, terms, concatenated):
        """Computes the observations for a given group.

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        3. Apply clipping based on :attr:`ObservationTermCfg.clip`
        4. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.
        """
        obs = dict()
        term: ObservationTerm
        for term in terms:
            this_obs: torch.Tensor = term.func(self.sim_data, self.robot_data, term.cfg).clone()
            if term.noise != None:
                this_obs = term.noise(this_obs)
            if term.clip > 0:
                this_obs.clip(min = -term.clip, max = term.clip)
            this_obs *= term.scale
            obs[term.func.__name__] = this_obs

        if concatenated:
            return torch.cat(list(obs.values()), dim=-1)
        else:
            return obs


    def reset(self, env_ids) -> Dict[str, torch.Tensor]:
        return {}

    """
    Helper functions.
    """

    def _prepare_groups(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), ObservationGroup)
                   and not name.startswith("__")]
        self.num_group = len(members)
        if self.num_group == 0:
            print("[WARNING]No term of type ObservationGroup. Env will return 'obs':tensor.")
            terms = self._prepare_terms(self)
            self.groups = []
            self.groups.append(("obs", terms))
            self.num_group = 1
        else:
            group: ObservationGroup
            self.groups = []
            for name in members:
                group = getattr(self.cfg, name)
                terms = self._prepare_terms(group)
                self.groups.append((name, terms, group.concatenate_terms))

    def _prepare_terms(self, group:ObservationGroup):
        attributes = dir(group)
        members = [name for name in attributes if
                   isinstance(getattr(group, name), ObservationTerm)
                   and not name.startswith("__")]
        num_term = len(members)
        if num_term == 0:
            return None
        terms = []
        for name in members:
            term = getattr(group, name)
            terms.append(term)
        return terms