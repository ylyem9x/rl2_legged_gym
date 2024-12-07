import numpy as np

def RobotDataWarpper(cls):
    def init(self):
        for term in self.robot_data_terms:
            num = term.init(self.sim_data, self)
            setattr(self, term.name, num)

    def reset(self, env_ids):
        extras = dict()
        for term in self.robot_data_terms:
            if term.reset != None:
                info = term.reset(self.sim_data, self, env_ids, getattr(self, term.name))
            extras.update(info)
        return extras

    def compute(self):
        for term in self.robot_data_terms:
            if term.compute != None:
                num = getattr(self, term.name)
                term.compute(self.sim_data, self, num)

    cls.init_term = init
    cls.reset_term = reset
    cls.compute_term = compute

    return cls