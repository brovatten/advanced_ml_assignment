import numpy as np


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = 0.05
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.random.uniform(size=(state_space, action_space))
        self.method = "Sarsa"

        self.last_state = None
        self.last_last_action = None
        self.last_action = self.choose_action(0)

    def observe(self, state, reward, done):
        last_q = self.Q[self.last_state, self.last_action]
        if done:
            self.Q[self.last_state, self.last_action] += self.alpha * (reward - last_q)
        else:
            self.Q[self.last_state, self.last_action] += self.alpha * (
                reward + self.gamma * self.Q[state, action] - last_q
            )
        self.last_action = action

    def choose_action(self, state):
        self.last_state = state
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.Q[state])

    def act(self, state):
        action = self.last_action
        self.last_last_action = self.last_action
        self.last_action = self.choose_action(state)
        self.last_state = state
        # We have already decided in __init__ or observe.
        return self.last_action
