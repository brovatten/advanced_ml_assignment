import sys
import argparse
import gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from plot_averages import ValueKeeper

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
    env = gym.make(args.env, is_slippery=False)
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
print(action_dim)
print(state_dim)

agent_type =  str(sys.argv[2])[:-3]
iterations = 20_000
alpha = 0.1
no_of_runs = 5
#print(agent_type)

def run_agent():
    agent = agentfile.Agent(state_dim, action_dim,alpha)

    observation = env.reset()
    keeper = ValueKeeper()

    for _ in range(iterations):
        action = agent.act(observation)  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        agent.observe(observation, reward, done)

        if done:
            keeper.add_value(reward)
            # print(agent.Q)
            observation = env.reset()

    return keeper


keepers = [run_agent() for _ in range(no_of_runs)]
episodes = min([len(x.values) for x in keepers])
averages = np.array([x.averages[:episodes] for x in keepers])
rewards = np.array([x.averages[:episodes] for x in keepers])
env.close()

#for i in range(len(rewards[0])):
    #print(rewards[:, i], "mean", rewards[:, i].mean(), "std", rewards[:, i].std())
mean = averages.mean(axis=0)
conf = 1.96 * rewards.std(axis=0)
plt.fill_between(range(len(mean)), mean - conf, mean + conf, alpha=0.2, color="grey", label = 'Confidence interval')
plt.title('Plot of ' + str(iterations) + ' steps of agent ' + str(agent_type))
plt.ylabel('Reward')
plt.xlabel('Episodes')
plt.plot(mean, label='Moving average (100)\n of rewards')
plt.legend()
plt.show()
plt.draw()
plt.close()
