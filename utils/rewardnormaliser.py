import numpy as np


class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, reward):
        self.count += 1
        last_mean = self.mean
        self.mean += (reward - self.mean) / self.count
        self.var += (reward - last_mean) * (reward - self.mean)
        std = np.sqrt(self.var / self.count)
        return reward / (std + self.epsilon)
