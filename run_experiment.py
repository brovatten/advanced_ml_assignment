import argparse
import gym
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import vis
from collections import deque
from tqdm import tqdm

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
    print(f"Training agent {k}")
    agent = agentfile.Agent(state_dim, action_dim)
    # print(vis.show(agent.Q))
    reward_queue = deque(maxlen=100)

    step = 0
    for i in tqdm(range(episodes)):
        done = False
        observation = env.reset()
        episode_reward = 0
        j = 0
        while True:
            j += 1
            step += 1
            # agent.Q = vis.A.copy()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.observe(observation, reward, done)
            episode_reward += reward
            if done:
                break

        reward_queue.append(episode_reward)
        rewards[k, i] = np.mean(reward_queue)

    print(f"Agent {k} done with {step} steps")
    # print(vis.show(agent.Q))


print(f"Known good Q")
print(vis.show(vis.A))

episodes = 20_000
rewards = np.zeros((2, episodes))
for k in range(2):
    run_agent(k)
env.close()

for i, xs in enumerate(rewards):
    plt.plot(xs, alpha=0.2, label=f"Agent {i}")
plt.plot(rewards.mean(axis=0), label="Average of all agents")
# plt.plot(0.01 + (1.0 - 0.01) * np.exp(-0.005 * np.arange(episodes)), label="Epsilon")
plt.legend()
plt.plot()
plt.show()
