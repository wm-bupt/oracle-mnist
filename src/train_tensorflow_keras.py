import argparse
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPool2D,Flatten
from tensorflow.keras.optimizers import Adam, SGD
import mnist_reader

def train(args):
    x_train, y_train = mnist_reader.load_data(args.data_dir, kind='train')
    x_test, y_test = mnist_reader.load_data(args.data_dir, kind='t10k')

    x_train = x_train.reshape(-1,1,28,28) / 255.
    x_test = x_test.reshape(-1,1,28,28) / 255.

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    model = Sequential()

    # Conv layer1
    model.add(Convolution2D(batch_input_shape=(None, 1, 28, 28), filters = 20, kernel_size = 5, strides = 1, padding = 'same', data_format = 'channels_first',))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size = 2, strides = 2, padding = 'same', data_format = 'channels_first',))
    # Conv layer2
    model.add(Convolution2D(50, 5, strides = 1, padding = 'same',data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(2,2,'same',data_format='channels_first'))
    # Fc
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr = args.lr, momentum=args.momentum)

    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print('Training------------Begin!!!')

    model.fit(x_train, y_train, epochs = args.epochs, batch_size = args.batch_size)
    print('Testing----------Begin!!!')

    loss, accuracy = model.evaluate(x_test, y_test)

    print('Oracle-MNIST accuracy: %.4f' % accuracy)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--data-dir', type=str, default='../data/oracle/', help='data path')

    args = parser.parse_args()
    train(args)

main()