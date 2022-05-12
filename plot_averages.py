import numpy as np
from collections import deque


class ValueKeeper:
    def __init__(self, moving_episodes=100):
        self.moving_episodes = moving_episodes
        self.last_values = deque()
        self.values = []
        self.averages = []

    def add_value(self, value):
        self.values.append(value)
        if len(self.last_values) >= self.moving_episodes:
            self.last_values.popleft()
        self.last_values.append(value)
        self.averages.append(np.mean(self.last_values))
