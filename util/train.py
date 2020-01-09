import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as torchnn
import torch.optim as optim
import os
import models.nn as nn
import models.cnn as cnn
import models.rnn as rnn
import models.lr as lr

MODEL_DIR = './out_model/'
MODEL_CNN_PATH = MODEL_DIR + 'cifar_cnn.pth'
MODEL_RNN_PATH = MODEL_DIR + 'cifar_rnn.pth'
MODEL_NN_PATH = MODEL_DIR + 'cifar_nn.pth'
MODEL_LR_PATH = MODEL_DIR + 'cifa_lr.pth'

def start_train(round_num=2, net_type=None):
    net = nn.Net()
    MODEL_PATH = MODEL_NN_PATH
    if net_type != None:
        if net_type == 'cnn':
            net = cnn.Net()
            MODEL_PATH = MODEL_CNN_PATH
        elif net_type == 'nn':
            net = nn.Net()
            MODEL_PATH = MODEL_NN_PATH
        elif net_type == 'lr':
            net = lr.lr()
            MODEL_PATH = MODEL_LR_PATH
        elif net_type == 'rnn':
            net = rnn.Net()
            MODEL_PATH = MODEL_RNN_PATH

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    # 训练集加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    # 图像数据的标签
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    criterion = torchnn.CrossEntropyLoss()

    # 优化算法，SGD随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    for epoch in range(round_num):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    if os.path.exists(MODEL_DIR) == False:
        os.makedirs(MODEL_DIR)
    torch.save(net.state_dict(), MODEL_PATH)
    return net
