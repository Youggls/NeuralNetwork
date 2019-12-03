from sklearn import svm
from sklearn.externals import joblib
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import os

MODEL_DIR = './out_model/'
MODEL_PATH = MODEL_DIR + 'cifar_svm.pkl'

def train_svm():
    model = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel='rbf')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    # 转为numpy数组，3*32*32转为一维
    data = list(trainset)
    images = list()
    labels = list()
    for image in data:
        images.append(image[0].view(-1, 3 * 32 * 32).numpy()[0, :])
        labels.append(image[1])
    images = numpy.array(images)[0:10000]
    labels = numpy.array(labels)[0:10000]
    # 图像数据的标签
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('Begin to fit svm model')
    model.fit(images, labels)
    print('Finish fiting svm model')
    if os.path.exists(MODEL_DIR) == False:
        os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    return model

def load_svm(path=None):
    if path != None:
        return joblib.load(path)
    return joblib.load(MODEL_PATH)

def test_svm(model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    # 转为numpy数组，3*32*32转为一维
    data = list(testset)
    images = list()
    labels = list()
    for image in data:
        images.append(image[0].view(-1, 3 * 32 * 32).numpy()[0, :])
        labels.append(image[1])
    images = numpy.array(images)
    labels = numpy.array(labels)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    predicts = model.predict(images)
    for (image, label, predict) in zip(images, labels, predicts):
        if label == predict:
            class_correct[label] += 1
        class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # Total Loss
    correct = sum(class_correct)
    total = sum(class_total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
