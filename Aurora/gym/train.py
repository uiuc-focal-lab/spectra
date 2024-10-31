# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import network_sim
import network_sim111
import torch
import random
import numpy as np
from model import CustomNetwork_mid, CustomNetwork_big, CustomNetwork_small

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default

K = 10

class MyMlpPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule=0.0001,
                 *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                         **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork_mid()

env = gym.make('PccNs-v0-pantheon')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO(MyMlpPolicy, env, seed=20, learning_rate=0.0001, verbose=1, batch_size=2048, n_steps=8192, gamma=gamma)

MODEL_PATH = f"./results/retrain_pcc_model_mid_{K}_%d.pt"
for i in range(0, K):
    model.learn(total_timesteps=(1600 * 410))

    torch.save(model.policy.state_dict(), MODEL_PATH % i)
