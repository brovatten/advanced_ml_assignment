import numpy as np

alpha = 0.01  # Learning rate
gamma = 0.99  # Discounting rate
epsilon = 0.05  # Exploration chance


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.Q = np.random.rand(state_space, action_space)

    def act(self, state):
        if np.random.random() > epsilon:
            self.a = np.argmax(self.Q[state, :])
        else:
            self.a = np.random.randint(self.action_space)
        self.s = state
        return self.a

    def observe(self, s_prime, r, done):
        if done:
            self.Q[s_prime, :] = 0
        error = r + gamma * np.max(self.Q[s_prime, :]) - self.Q[self.s, self.a]
        self.Q[self.s, self.a] += alpha * error
