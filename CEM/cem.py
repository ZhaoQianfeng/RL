""" Implemetation of Cross Entropy Method 
@author: ZhaoQianfeng
"""
import numpy as np



def rollout(net, env, n_steps, n_iter):
    """rollout to evaluate the score of parameters
    
    net: the net which decides action
    env: the environment
    n_steps: the number of max_step of a episode
    n_iter: the number of episodess
    """
    rewards = [0]*n_iter
    for i in range(n_iter): 
        observation = env.reset()
        for step in range(n_steps):
            action = net(observation)
            observation, reward, done, _info = env.step(action)
            rewards[i] += reward
            if done:
                break
    avg_reward = sum(rewards)/len(rewards)
    
    return avg_reward


class BinaryDecisionNet:
    """A linear model whose output decide the agent action"""
    def __init__(self, theta, n_action):
        self.theta = theta
        theta_matrix = self.theta.reshape(-1,n_action)
        self.w = theta_matrix[:-1,:]
        self.b = theta_matrix[-1,:]
    
    def __call__(self, x):
        out = x.dot(self.w) + self.b

        return np.argmax(out)
    

def cem(eva_func, env,old_mean, old_std, n_models, len_theta, select_ratio=0.2,
        n_steps=200, n_iter=10,n_action=2):
    """implemetation of cem
    
    eva_func: the func give the score of a net
    old_mean: the mean of parameters, shape is (theta,)
    old_std: the standard deviation of parameters, shape is (theta,)
    n_models: number of generated models
    select_ratio: select top ratio model
    """

    ## generate `n_models` models, following Gaussian Distribution
    models_param = [old_mean+std for std in np.random.randn(n_models, len_theta)*old_std]
    models = [BinaryDecisionNet(param,n_action) for param in models_param]

    ## caculate scores of models
    scores = [rollout(model, env, n_steps, n_iter ) for model in models]
    scores = np.array(scores)
    ## select models
    n_select_models = int(round(n_models*select_ratio))
    top_models_idx = scores.argsort()[::-1][:n_select_models]
    selected_models_params = []
    print(scores[top_models_idx[0]])
    for idx in top_models_idx:
        selected_models_params.append(models[idx].theta)
    thetas = np.stack(selected_models_params,axis=0)
    
    ## calculate mean and std of the selected models
    mean = np.mean(thetas,axis=0)
    std = np.std(thetas,axis=0)
    std = std + np.random.normal(0,0.01)
    ## return the new mean and new std
    return mean, std, models[top_models_idx[0]]





    
    

    

