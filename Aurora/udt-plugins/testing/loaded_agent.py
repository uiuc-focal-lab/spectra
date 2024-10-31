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

import numpy as np
import io
import torch

K=10

class CustomNetwork(torch.nn.Module):
    def __init__(self, feature_dim: int = K*3,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, last_layer_dim_pi),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, last_layer_dim_vf),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features)

    def forward_actor(self, features):
        return self.policy_net(features)



class LoadedModel():

    def __init__(self, model_path):
        self.model_path = model_path

        self.initial_state = np.zeros((K, 3), dtype=np.float32)
        self.state = np.zeros((K, 3), dtype=np.float32)

        self.model = CustomNetwork()
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()

    def reset_state(self):
        self.state = np.copy(self.initial_state)

    def reload(self):
        self.model = CustomNetwork()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def act(self, obs, stochastic=False):
        obs = torch.from_numpy(obs).to(torch.float32)
        sess_output = self.model.forward(obs)

        action = None
        if len(sess_output) > 1:
            action, self.state = sess_output
        else:
            action = sess_output.detach().numpy()
        
        return {"act": action}


class LoadedModelAgent():
    def __init__(self, model_path):
        self.model = LoadedModel(model_path)

    def reset(self):
        self.model.reset_state()

    def act(self, ob):
        act_dict = self.model.act(ob.reshape(1, -1), stochastic=False)
        act = act_dict["act"]
        vpred = act_dict["vpred"] if "vpred" in act_dict.keys() else None
        state = act_dict["state"] if "state" in act_dict.keys() else None
        return act[0][0]
