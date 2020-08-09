import torch.nn as nn
import torch
import numpy as np

class Net(nn.Module):
    # @property
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28*28, 1024, bias=True)
        self.fun1=nn.ReLU()
        self.linear2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        variable1 = self.linear1(x)
        out=self.fun1(variable1)
        variable2=self.linear2(out)
        out_sm = self.softmax(variable2)
        return variable2, out_sm

class Model:
    def __init__(self, name):
        self.name = name
        self.net = Net()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def Loss(self, x, y_true):
        x = torch.FloatTensor(np.array(x))
        y_true = torch.FloatTensor(np.array(y_true))
        y_pred, y_pred_sm = self.net(x)
        y_pred = torch.FloatTensor(y_pred)
        loss = self.loss_func(y_pred, torch.max(y_true, 1)[1])
        return loss

    def Train(self, train_x, train_y):
        L = self.Loss(train_x, train_y)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()