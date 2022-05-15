import numpy as np


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = 0.05
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = (
            np.random.uniform(size=(state_space, action_space)),
            np.random.uniform(size=(state_space, action_space)),
        )
        self.method = "Double-Q"
        self.last_action = None
        self.last_state = None

    def observe(self, state, reward, done):
        a = np.random.randint(2)  # Decide which of the Q matrices we are using.
        b = 1 - a
        last_q = self.Q[a][self.last_state, self.last_action]
        if done:
            self.Q[a][self.last_state, self.last_action] += self.alpha * (
                reward - last_q
            )
        else:
            self.Q[a][self.last_state, self.last_action] += self.alpha * (
                reward
                + self.gamma * self.Q[b][state, np.argmax(self.Q[a][state])]
                - last_q
            )

    def act(self, state):
        self.last_state = state
        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.action_space)
        else:
            self.last_action = np.argmax(self.Q[0][state] + self.Q[1][state])
        return self.last_action
