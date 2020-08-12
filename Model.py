import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    # @property
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 1024, bias=True)
        self.fun1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        variable1 = self.linear1(x)
        out = self.fun1(variable1)
        variable2 = self.linear2(out)
        return variable2


class Model:
    def __init__(self, name, device):
        self.name = name
        self.net = Net()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        if device != "cpu":
            self.net.cuda(self.device)

    def loss(self, x, y_true):
        x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        y_true = torch.tensor(np.array(y_true), dtype=torch.float32, device=self.device)
        y_pred = self.net(x)
        loss = self.loss_func(y_pred, torch.max(y_true, 1)[1])
        return loss

    def train(self, train_x, train_y):
        L = self.loss(train_x, train_y)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()

    def inference(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        return self.net(x).cpu().detach().numpy()
