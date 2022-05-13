import numpy as np
import random

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP


class Agent(object):  # Keep the class name!
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space, alpha=0.7):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = (np.random.uniform(size=(state_space, action_space)),
        np.random.uniform(size=(state_space, action_space)))
        #print(self.Q)
        #self.Q = (
        #    np.ones((state_space, action_space)),
        #    np.ones((state_space, action_space)),
        #)
        #self.Q[0][15, :] = 0
        #self.Q[1][15, :] = 0
        self.method = "Double-Q"
        self.state_n = (0, 0)
        self.last_action = -1
        self.last_state = 0

    def argmax_action(self, q, state):
        if np.array_equal(q[state, :], [0.0, 0.0, 0.0, 0.0]):
            return np.random.randint(self.action_space)
        return np.argmax(q[state, :])

    def observe(self, state, reward, done):
        a = random.randint(0, 1)
        b = 1 - a
        self.Q[a][self.last_state, self.last_action] += self.alpha * (
            reward
            + self.gamma * self.Q[b][state, self.argmax_action(self.Q[a], state)]
            - self.Q[a][self.last_state, self.last_action]
        )

        if done:
            self.Q[0][state,:] = 0 
            self.Q[1][state,:] = 0

    def act(self, state):
        self.last_state = state
        random_action = np.random.randint(self.action_space)
        max_action = self.argmax_action(self.Q[0] + self.Q[1], state)
        action = np.random.choice(
            [max_action, random_action], p=[1 - self.epsilon, self.epsilon]
        )
        self.last_action = action
        return action
