import argparse
import gym
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import vis
from collections import deque
from tqdm import tqdm
import scipy.stats as st

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
    reward_queue = deque(maxlen=1000)

    step = 0
    for i in tqdm(range(episodes)):
        done = False
        observation = env.reset()
        episode_reward = 0
        while True:
            step += 1
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.observe(observation, reward, done)
            episode_reward += reward
            if done:
                break

        reward_queue.append(episode_reward)
        rewards[k, i] = np.mean(reward_queue)

    print(f"Agent {1+k} done with {step} steps")


episodes = 10_000
agents = 5

rewards = np.zeros((agents, episodes))
for k in range(agents):
    run_agent(k)
env.close()

mean = rewards.mean(axis=0)
conf = np.array(
    [
        st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
        for a in rewards.T
    ]
)
plt.fill_between(
    range(len(mean)),
    conf[:, 0],
    conf[:, 1],
    alpha=0.2,
    color="grey",
    label=r"95% confidence interval",
)
for i, xs in enumerate(rewards):
    plt.plot(xs, alpha=0.2, label=f"Agent {1+i}")
plt.plot(mean, label="Average of all agents")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.plot()
plt.show()
