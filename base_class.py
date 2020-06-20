from block_utils import ZERO_ROT, ZERO_POS


class ActionBase:
    def __init__(self, T=5):
        self.rot = ZERO_ROT
        self.pos = ZERO_POS
        self.T = T

    def step(self):
        pass


class PlannerBase:
    def __init__(self):
        pass

    def plan(self):
        pass


class BeliefBase:
    def __init__(self):
        pass

    def update(self, observation):
        pass