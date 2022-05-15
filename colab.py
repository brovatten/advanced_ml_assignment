import numpy as np
import random

learning_rate = 0.7  # Learning rate
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob


class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.episode = 0
        self.epsilon = 1.0
        self.state = -1
        self.action = -1
        self.Q = np.zeros((state_space, action_space))

    def act(self, state):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > self.epsilon:
            action = np.argmax(self.Q[state, :])
        else:
            action = np.random.randint(self.action_space)
        self.state = state
        self.action = action
        return action

    def observe(self, new_state, reward, done):
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        self.Q[self.state, self.action] = self.Q[
            self.state, self.action
        ] + learning_rate * (
            reward
            + gamma * np.max(self.Q[new_state, :])
            - self.Q[self.state, self.action]
        )

        if done:
            self.episode += 1
            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay_rate * self.episode
            )
            # print(self.epsilon)
