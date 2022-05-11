import argparse
import gym
import importlib.util
import time
import plot_averages

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


agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()
plot = plot_averages.plot_averages()
for _ in range(10000):
    # env.render()
    action = agent.act(observation)  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    agent.observe(observation, reward, done)

    if done:
        plot.add_value(reward)
        print(agent.Q)
        observation = env.reset()
plot.plot_averages()
env.close()
