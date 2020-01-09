from models import lr
from util import train, test
if __name__ == '__main__':
    lr = train.start_train(net_type='lr')
    # rnn_net = rnn.load_net('./out_model/cifar_cnn.pth')
    test.test_nn(lr)
    
    # net = train.start_train(net_type='nn')
    # test.test_nn(net)
