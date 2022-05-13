import numpy as np

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP


class Agent(object):  # Keep the class name!
    """Sarsa agent"""

    def __init__(self, state_space, action_space, alpha=0.7):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.zeros((state_space, action_space))
        self.state_n = (0, 0)
        self.last_state = 0

        self.last_action = self.choose_action(0)

    def observe(self, state, reward, done):
        action = self.choose_action(state)
        self.Q[self.last_state][self.last_action] += self.alpha * (
            reward
            + self.gamma * self.Q[state][action]
            - self.Q[self.last_state][self.last_action]
        )
        self.last_action = action
        if done: 
            self.Q[state,:] = 0 

    def choose_action(self, state):
        random_action = np.random.randint(self.action_space)
        max_action = np.argmax(self.Q[state, :])
        action = np.random.choice(
            [max_action, random_action], p=[1 - self.epsilon, self.epsilon]
        )
        if np.array_equal(self.Q[state, :], [0.0, 0.0, 0.0, 0.0]):
            action = np.random.randint(self.action_space)
        return action

    def act(self, state):
        self.last_state = state
        return self.last_action
