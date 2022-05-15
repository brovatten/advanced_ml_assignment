import numpy as np


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.random.uniform(size=(state_space, action_space))
        self.method = "Sarsa"

        self.last_state = None
        self.last_action = None
        self.last_last_state = None
        self.last_last_action = None
        self.last_reward = None

    def observe(self, state, reward, done):
        if self.last_last_action is not None or self.last_last_state is not None:
            # Observe the result of the action we took (last_action) with the respect
            # to the one before that (last_last_action).
            last_q = self.Q[self.last_state, self.last_action]
            last_last_q = self.Q[self.last_last_state, self.last_last_action]
            self.Q[self.last_last_state, self.last_last_action] += self.alpha * (
                self.last_reward + self.gamma * last_q - last_last_q
            )

        if done:
            # Do the final update - the current state is terminal so we will never get
            # a chance to observe again.
            self.Q[self.last_state, self.last_action] += self.alpha * (reward - last_q)

        self.last_reward = reward

    def act(self, state):
        self.last_last_state = self.last_state
        self.last_last_action = self.last_action
        self.last_state = state
        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.action_space)
        else:
            self.last_action = np.argmax(self.Q[state])
        return self.last_action
