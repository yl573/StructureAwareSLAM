import numpy as np


class CompositeLossTracker:

    def __init__(self, losses):
        """
        :param losses: list of string loss names
        """
        self.losses = {}
        for k in losses:
            self.losses[k] = LossTracker()

    def update(self, loss_dict):
        """
        :param loss_dict: {'loss_name', loss_val, ...}
        """
        for k, v in loss_dict.items():
            self.losses[k].update(v)

    def mean(self):
        return {k: l.mean() for k, l in self.losses.items()}

    def reset(self):
        for k in self.losses:
            self.losses[k].reset()


class LossTracker:

    def __init__(self):
        self.losses = []

    def update(self, loss):
        self.losses.append(loss)

    def mean(self):
        return np.mean(self.losses)

    def reset(self):
        self.losses = []
