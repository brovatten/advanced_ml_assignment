import numpy as np

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP


class Agent(object):  # Keep the class name!
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = 0.7
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.zeros((state_space, action_space))
        self.method = "Qlearning"
        self.state_n = (0, 0)
        self.last_action = -1
        self.last_state = 0

    def observe(self, observation, reward, done):
        self.Q[self.last_state][self.last_action] += self.alpha * (
            reward
            + self.gamma * max(self.Q[observation][:])
            - self.Q[self.last_state][self.last_action]
        )

    def act(self, observation):
        return self.Qlearning(observation)

    def Qlearning(self, observation):
        self.last_state = observation
        random_action = np.random.randint(self.action_space)
        max_action = np.argmax(self.Q[observation, :])
        action = np.random.choice(
            [max_action, random_action], p=[1 - self.epsilon, self.epsilon]
        )
        if np.array_equal(self.Q[observation, :], [0.0, 0.0, 0.0, 0.0]):
            action = np.random.randint(self.action_space)
        self.last_action = action
        return action
