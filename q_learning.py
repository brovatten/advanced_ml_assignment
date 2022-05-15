import numpy as np


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = 0.05
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.random.uniform(size=(state_space, action_space))
        self.method = "Q-Learning"
        self.last_action = None
        self.last_state = None

    def observe(self, state, reward, done):
        last_q = self.Q[self.last_state, self.last_action]
        if done:
            # Don't include the terminal state in the error.
            self.Q[self.last_state, self.last_action] += self.alpha * (reward - last_q)
        else:
            self.Q[self.last_state, self.last_action] += self.alpha * (
                reward + self.gamma * np.max(self.Q[state]) - last_q
            )

    def act(self, state):
        self.last_state = state
        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.action_space)
        else:
            self.last_action = np.argmax(self.Q[state])
        return self.last_action
