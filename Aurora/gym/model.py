import gym
import torch
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from common.simple_arg_parse import arg_or_default


K=10

class CustomNetwork_mid(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

class CustomNetwork_mid_policy_net(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        # print('features:', features)
        # for layer in self.policy_net:
        #     features = layer(features)
        #     print('layer:', layer, 'features:', features) 
        return self.policy_net(features)


class CustomNetwork_big(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomNetwork_small(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)
    
class CustomNetwork_small_policy_net(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )
        

    def forward(self, features):
        return self.policy_net(features)