from models import cnn

if __name__ == '__main__':
    cnn_net = cnn.load_net('./out_model/cifar_cnn.pth')
    cnn.test_cnn(cnn_net)
