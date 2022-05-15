import numpy as np

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP

TERMS = {5, 7, 11, 12, 15}


class Agent(object):  # Keep the class name!
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space, alpha=0.05):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = 0.95
        self.epsilon = 0.05
        self.Q = np.random.random((state_space, action_space))
        self.method = "Qlearning"
        self.state_n = (0, 0)
        self.last_action = -1
        self.last_state = 0
        self.step = 0

    def observe(self, state, reward, done):
        # self.Q[self.last_state][self.last_action] += self.alpha * (
        #     reward
        #     + self.gamma * max(self.Q[state][:])
        #     - self.Q[self.last_state][self.last_action]
        # )

        if done:
            assert self.last_state not in TERMS
            print(state, self.step)
            self.step = 0
            assert state in TERMS
            # self.Q[state, :] = 0.0  # ?????????
            self.Q[self.last_state][self.last_action] += self.alpha * (
                reward
                # + self.gamma * max(self.Q[state][:])
                - self.Q[self.last_state][self.last_action]
            )
        else:
            self.step += 1
            assert self.last_state not in TERMS
            assert state not in TERMS
            self.Q[self.last_state][self.last_action] += self.alpha * (
                reward
                + self.gamma * max(self.Q[state][:])
                - self.Q[self.last_state][self.last_action]
            )

    def act(self, observation):
        return self.Qlearning(observation)

    def Qlearning(self, observation):
        assert observation not in TERMS
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
