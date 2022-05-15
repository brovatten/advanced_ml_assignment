import gym
import numpy as np
from collections import deque
import vis
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)
Q2 = np.random.random([env.observation_space.n, env.action_space.n])
Q1 = np.random.random([env.observation_space.n, env.action_space.n])
learning_rate = 0.05
discount = 0.99
num_episodes = 20_000
rList = deque(maxlen=100)
avgList = []

for i in range(num_episodes):
    s = env.reset()
    done = False
    j = 0
    while j < 99:
        j += 1
        if np.random.random() < 0.05:
            a = np.random.randint(env.action_space.n)
        else:
            a = np.argmax(Q1[s] + Q2[s])
        new_s, reward, done, _ = env.step(a)
        if np.random.uniform() < 0.5:
            Q1[s, a] += learning_rate * (
                reward + discount * Q2[new_s, np.argmax(Q1[new_s, :])] - Q1[s, a]
            )
        else:
            Q2[s, a] += learning_rate * (
                reward + discount * Q1[new_s, np.argmax(Q2[new_s, :])] - Q2[s, a]
            )
        s = new_s
        if done:
            Q1[new_s, :] = 0
            Q2[new_s, :] = 0
            rList.append(reward)
            avgList.append(np.mean(rList))
            break

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q1)
print(vis.show(Q1))
print(Q2)
print(vis.show(Q2))
plt.plot(avgList)
plt.show()
