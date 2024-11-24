import torch
import numpy as np
from typing import Dict
from prettytable import PrettyTable
from legged_gym.data import SimData, RobotData
from legged_gym.managers.manager_term_cfg import ActionComputeTerm, ActionSacleTerm

class ActionManager:
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    def __init__(self, cfg, sim_data: SimData, robot_data: RobotData):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
        """
        self.cfg = cfg
        self.sim_data = sim_data
        self.robot_data = robot_data
        self.action_scale = None
        self.clip_params = None
        self.control_type = None
        self.terms = self._prepare_terms()

    def __str__(self) -> str:
        """Returns: A string representation for action scale."""
        msg = f"<ActionManager> contains {self.action_scale.shape[0]} action term.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Action scale Terms"
        table.field_names = ["dof id", "scale", "clip params"]
        # set alignment of table columns
        table.align["dof id"] = "l"
        # add info on each term
        for i in range(self.sim_data.num_actions):
            table.add_row([i, self.action_scale[i].detach().cpu().numpy(), 
                           self.clip_params[i].detach().cpu().numpy()])
        # convert table to string
        msg += table.get_string()
        msg += "\n"
        msg += f"Control type is {self.control_type}."
        if self.control_type == "p":
            msg += f"kp = {self.kp}, kd = {self.kd}"
        elif self.control_type == "other":
            msg += f"Control function is {self.compute_func.__name__}"
        msg += "\n"
        return msg

    def process_action(self, action: torch.Tensor):
        # return cliped action
        # scale is not applied here because the cliped action is observation
        if action.shape[1] == self.sim_data.num_actions:
            action_cliped = torch.clip(action, -self.clip_params, self.clip_params)
        else:
            raise ValueError(f"Invaild action shape, expected:{self.sim_data.num_actions}, recevied:{action.shape[1]}")
        return action_cliped

    def compute_torque(self, action: torch.Tensor):
        action_scaled = action * self.action_scale

        if self.control_type == "P":
            # self.robot_data.lag_buffer = self.robot_data.lag_buffer[1:] + [action_scaled.clone()]
            # self.joint_pos_target = self.robot_data.lag_buffer[0] + self.sim_data.default_dof_pos
            self.joint_pos_target = action_scaled + self.robot_data.default_dof_pos
            torque = self.robot_data.p_gain * self.kp * (
                    self.joint_pos_target - self.robot_data.dof_pos + self.robot_data.motor_offset) \
                    - self.robot_data.d_gain * self.kd * self.robot_data.dof_vel
        elif self.control_type == "T":
            self.robot_data.lag_buffer = self.robot_data.lag_buffer[1:] + [action_scaled.clone()]
            torque = self.robot_data.lag_buffer[0]
        elif self.control_type == "other":
            torque = self.compute_func(self.sim_data, self.robot_data, action_scaled)

        torque = torque * self.robot_data.motor_strength
        return torque

    """
    Helper functions.
    """

    def _prepare_terms(self):
        attributes = dir(self.cfg)
        members = [name for name in attributes if
                   isinstance(getattr(self.cfg, name), ActionSacleTerm)
                   and not name.startswith("__")]
        if len(members) != 1:
            raise ValueError(f"It has {len(members)} ActionScaleTerm, but expected 1 term.")
        term: ActionSacleTerm = getattr(self.cfg, members[0])

        self.action_scale = self._prepare_terms_data_into_tensor(term.scale, "Scale")
        self.clip_params = self._prepare_terms_data_into_tensor(term.clip, "Clip")

        compute_members = [name for name in attributes if
                            isinstance(getattr(self.cfg, name), ActionComputeTerm)
                            and not name.startswith("__")]
        if len(compute_members) != 1:
            raise ValueError(f"ActionComputeTerm has {len(compute_members)}, expected 1.")
        compute_term: ActionComputeTerm = getattr(self.cfg, compute_members[0])

        self.control_type = compute_term.control_type
        if compute_term.control_type == "P":
            self.kp = self._prepare_terms_data_into_tensor(compute_term.kp, "Kp")
            self.kd = self._prepare_terms_data_into_tensor(compute_term.kd, "Kd")
        elif compute_term.control_type == "other":
            self.compute_func = compute_term.func
        elif compute_term.control_type != "T":
            raise ValueError(f"invalid control type in ActionComputeTerm, expected P/T/other.")

    def _prepare_terms_data_into_tensor(self, data, term_name) -> torch.Tensor:
        """process term data. A number or list is available."""
        if isinstance(data, float) or isinstance(data, int):
            tensor = torch.ones(self.sim_data.num_actions, device=self.sim_data.device,
                                dtype=torch.float32) * data
        elif len(data) == self.sim_data.num_actions:
            tensor = torch.from_numpy(np.array(data)).to(self.sim_data.device)
        else:
            raise ValueError(f"The {term_name} Params has wrong type or size, expected int/float/list({self.sim_data.num_actions}), received: {data}.")
        return tensor
