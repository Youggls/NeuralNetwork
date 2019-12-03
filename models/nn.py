import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SIZE = WIDTH * HEIGHT * CHANNELS
MODEL_PATH = './out_model/cifar_nn.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(SIZE, 100)
        self.layer2 = nn.Linear(100, 30)
        self.layer3 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, SIZE)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.softmax(x, dim=1)
        return x

def load_net(path=None):
    net = Net()
    if path == None:
        net.load_state_dict(torch.load(MODEL_PATH))
    else:
        net.load_state_dict(torch.load(path))
    return net
