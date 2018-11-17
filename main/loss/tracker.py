import numpy as np


class LossTracker:

    def __init__(self):
        self.losses = []

    def update(self, loss):
        self.losses.append(loss)

    def mean(self):
        return np.mean(self.losses)

    def reset(self):
        self.losses = 0
