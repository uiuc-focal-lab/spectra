import torch

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

K = 10


class CustomNetwork_mid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        y = self.policy_net(features)
        return y


class CustomNetwork_mid_parallel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        x1, x2 = torch.split(features, 30, dim=1)
        y1 = self.policy_net(x1)
        y2 = self.policy_net(x2)
        # Marabou does not support sub
        y2 = torch.mul(y2, -1)
        ret = torch.add(y1, y2)
        return ret


class CustomNetwork_mid_concatnate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):
        print(features.size())
        x1, x2, x3, x4, x5, constant = torch.split(features, 30, dim=1)

        y = self.policy_net(x1)
        y = torch.mul(y, 0.025)
        ret = torch.add(y, 1)

        y1, y2, y3 = torch.split(x2, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x2 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x2)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x3, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x3 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x3)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x4, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x4 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x4)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x5, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x5 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x5)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        return ret


class CustomNetwork_big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        y = self.policy_net(features)
        return y


class CustomNetwork_big_parallel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        x1, x2 = torch.split(features, 30, dim=1)
        y1 = self.policy_net(x1)
        y2 = self.policy_net(x2)
        # Marabou does not support sub
        y2 = torch.mul(y2, -1)
        ret = torch.add(y1, y2)
        return ret


class CustomNetwork_big_concatnate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):
        print(features.size())
        x1, x2, x3, x4, x5, constant = torch.split(features, 30, dim=1)

        y = self.policy_net(x1)
        y = torch.mul(y, 0.025)
        ret = torch.add(y, 1)

        y1, y2, y3 = torch.split(x2, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x2 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x2)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x3, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x3 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x3)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x4, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x4 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x4)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x5, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x5 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x5)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        return ret


class CustomNetwork_small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        y = self.policy_net(features)
        return y


class CustomNetwork_small_parallel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        x1, x2 = torch.split(features, 30, dim=1)
        y1 = self.policy_net(x1)
        y2 = self.policy_net(x2)
        # Marabou does not support sub
        y2 = torch.mul(y2, -1)
        ret = torch.add(y1, y2)
        return ret


class CustomNetwork_small_concatnate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):
        print(features.size())
        x1, x2, x3, x4, x5, constant = torch.split(features, 30, dim=1)

        y = self.policy_net(x1)
        y = torch.mul(y, 0.025)
        ret = torch.add(y, 1)

        y1, y2, y3 = torch.split(x2, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x2 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x2)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x3, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x3 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x3)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x4, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x4 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x4)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y1, y2, y3 = torch.split(x5, [19, 1, 10], dim=1)
        y2 = torch.add(y2, y.detach())
        x5 = torch.concat([y1, y2, y3], dim=1)

        y = self.policy_net(x5)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        return ret
