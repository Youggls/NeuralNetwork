from models import cnn
from util import train, test
if __name__ == '__main__':
    net = train.start_train(net_type='nn')
    # cnn_net = cnn.load_net('./out_model/cifar_cnn.pth')
    test.test_nn(net)
