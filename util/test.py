import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as torchnn
import torch.optim as optim
import models.nn as nn
import models.cnn as cnn
import models.rnn as rnn


def test_nn(net):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    # 测试集加载器
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

        # 图像数据的标签
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # Total Loss
    correct = sum(class_correct)
    total = sum(class_total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
