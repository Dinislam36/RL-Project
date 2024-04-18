import numpy as np
import torch


class CustomScheduler:
    def __init__(self, optim):
        self.optim = optim
        self.initial_lr = optim.param_groups[0]['lr']
        self.last_reward = None
        self.same_reward_streak = 0
        self.sigmoid = lambda x: 1 / (1 + np.exp(-0.4*x)) - 0.3

    def step(self, reward):
        if reward == self.last_reward:
            self.optim.param_groups[0]['lr'] *= self.sigmoid(self.same_reward_streak)
            self.same_reward_streak += 1
        else:
            self.optim.param_groups[0]['lr'] = self.initial_lr
            self.same_reward_streak = 0
