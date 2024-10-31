
import gym
import network_sim
import torch
import random
import numpy as np
from model import CustomNetwork_mid, CustomNetwork_big, CustomNetwork_small
import argparse
import json
import pandas as pd
import sys 
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('-save_df', type=str, default=None)
args = parser.parse_args()

env = gym.make('PccNs-v0-pantheon')

@torch.no_grad()
def test(model, save_df=None):
    model.eval()
    test_scores = []
    states = []
    actions = []
    for j in range(605):

        state, d, test_score = env.reset(), False, 0

        state = state.astype('float32')
        i = 0
        while not d:

            action, _ = model.forward(torch.tensor(state))
            states += [state]
            actions += [action]
            state, r, d, _ = env.step(action)
            i += 1
            state = state.astype('float32')
            test_score += r
        test_scores.append(test_score)
    if save_df is not None:
        df = pd.DataFrame(states)
        df['actions'] = actions
        df.to_csv(save_df, index=False)
    return test_scores


random.seed(0)
np.random.seed(0)


best = 0
best_reward = -100
reward_list = []
for i in range(1):
    model_path = args.model_path
    model = CustomNetwork_mid()

    state_dict = torch.load(model_path)

    for key in list(state_dict.keys()):
        state_dict[key.replace('mlp_extractor.', '')] = state_dict.pop(key)

    state_dict.requires_grad = False
    model.load_state_dict(state_dict, strict=False)
    
    reward2 = test(model, args.save_df)
    # reward1 = test(model)
    sum_tensor = sum(reward2)
    sum_tensor_double = sum_tensor.double()/len(reward2)
    print(sum_tensor_double)
    reward_list.append(sum_tensor_double)
    if sum_tensor_double>best_reward:
        best = i
        best_reward = sum_tensor_double

print(reward_list)
print(f"best: {best}")