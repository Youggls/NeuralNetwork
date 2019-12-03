import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SIZE = WIDTH * HEIGHT * CHANNELS
MODEL_PATH = './out_model/cifar_lr.pth'

class lr(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(SIZE, 10)

    def forward(self, x):
        x = x.view(-1, SIZE)
        x = self.weight(x)
        return F.sigmoid(x)
def load_net(path=None):
    net = lr()
    if path == None:
        net.load_state_dict(torch.load(MODEL_PATH))
    else:
        net.load_state_dict(torch.load(path))
    return net