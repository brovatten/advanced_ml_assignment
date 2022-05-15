import numpy as np

learning_rate = 0.05
discount = 0.99
num_episodes = 20_000


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.Q2 = np.random.random([state_space, action_space])
        self.Q1 = np.random.random([state_space, action_space])

    def act(self, s):
        if np.random.random() < 0.05:
            self.a = np.random.randint(self.action_space)
        else:
            self.a = np.argmax(self.Q1[s] + self.Q2[s])
        self.s = s
        return self.a

    def observe(self, new_s, reward, done):
        s = self.s
        a = self.a
        if np.random.uniform() < 0.5:
            self.Q1[s, a] += learning_rate * (
                reward
                + discount * self.Q2[new_s, np.argmax(self.Q1[new_s, :])]
                - self.Q1[s, a]
            )
        else:
            self.Q2[s, a] += learning_rate * (
                reward
                + discount * self.Q1[new_s, np.argmax(self.Q2[new_s, :])]
                - self.Q2[s, a]
            )

        if done:
            self.Q2[new_s, :] = 0
            self.Q1[new_s, :] = 0
