import random

import gym
import torch
import torch.nn as nn
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float().unsqueeze(0)
    elif env in ['Pong-v0']:
        env_config = config.Pong
        obs = torch.tensor(obs, device=device).float()
        obs = obs.unsqueeze(0)
        #updating the frame stack
        obs_stack  =  torch.cat(env_config["obs_stack_size"]  * [obs]).unsqueeze(0).to(device)
        obs = torch.cat((obs_stack[:,1:, ...],obs.unsqueeze(1)), dim = 1 ).to(device)
        return obs
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
