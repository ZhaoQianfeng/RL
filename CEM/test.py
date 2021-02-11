from operator import mul
from functools import reduce


import gym
from cem import *
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('env', nargs='?', default='CartPole-v0')
opt = parser.parse_args()


env = gym.make(opt.env)


N_ACTION = env.action_space.n
len_theta = env.reset().shape
len_theta = (reduce(mul, len_theta) + 1) * N_ACTION

# initialize mean and std
mean = np.zeros(len_theta)
init_std = 0.1
std = np.ones_like(mean) * init_std

N_MODELS = 50
N_UPDATE = 10

for update in range(N_UPDATE):
    mean, std, top_model = cem(rollout, env, mean,std, N_MODELS, len_theta, n_action=N_ACTION) 

# model_param = mean + np.random.randn(len_theta) * std
agent = top_model

# Test the agent
if __name__ == '__main__'
    N_EPISODE = 50
    N_STEP = 5000
    for _ in range(N_EPISODE):
        observation = env.reset()
        env.render()
        for step in range(N_STEP):
            action = agent(observation)
            observation, reward, done, _info = env.step(action)
            env.render()
            time.sleep(0.01)
            if done:
                print(f'STEP {step}')
                break



