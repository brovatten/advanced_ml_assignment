import argparse
import gym
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import vis
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agentfile", type=str, help="file with Agent object", default="agent.py"
)
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("Agent", args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env, is_slippery=True)
    print("Loaded ", args.env)
except:
    print(args.env + ":Env")
    gym.envs.register(
        id=args.env + "-v0",
        entry_point=args.env + ":Env",
    )
    env = gym.make(args.env + "-v0")
    print("Loaded", args.env)

action_dim = env.action_space.n
state_dim = env.observation_space.n


def run_agent(k):
    agent = agentfile.Agent(state_dim, action_dim)
    reward_queue = deque(maxlen=100)

    step = 0
    for i in range(episodes):
        done = False
        observation = env.reset()
        episode_reward = 0
        while not done:
            step += 1
            # agent.Q = vis.A.copy()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.observe(observation, reward, done)
            episode_reward += reward

        reward_queue.append(episode_reward)
        rewards[k, i] = np.mean(reward_queue)

    print(f"Agent {k} done with {step} steps")


episodes = 15_000
rewards = np.zeros((5, episodes))
for k in range(5):
    run_agent(k)
env.close()

for i, xs in enumerate(rewards):
    plt.plot(xs, alpha=0.2, label=f"Agent {i}")
plt.plot(rewards.mean(axis=0), label="Average of all agents")
plt.legend()
plt.plot()
plt.show()
