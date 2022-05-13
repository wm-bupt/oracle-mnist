import argparse
import tensorflow as tf
import mnist_reader_tf as mnist_reader

def train(args):
  mnist = mnist_reader.read_data_sets(args.data_dir, one_hot=True, valid_num=args.valid_num)

  print('The size of training data: %d' % mnist.train.labels.shape[0])
  print('The size of test data: %d' % mnist.test.labels.shape[0])
  if args.valid_num != 0:
    print('The size of validation data: %d' % mnist.validation.labels.shape[0])

  sess = tf.InteractiveSession()
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

  y = tf.nn.softmax(tf.matmul(x,W) + b)

  y_ = tf.placeholder(tf.float32, [None,10])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(cross_entropy)


  tf.global_variables_initializer().run()

  for i in range(args.iter):
    batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
    train_step.run({x: batch_xs, y_: batch_ys})


  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print('Oracle-MNIST ACCURACY: %.4f' % accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


def main():
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
  parser.add_argument('--iter', type=int, default=1000, metavar='N', help='number of iterations to train (default: 1000)')
  parser.add_argument('--lr', type=float, default=0.5, metavar='LR', help='learning rate (default: 0.5)')
  parser.add_argument('--data-dir', type=str, default='../data/oracle/', help='data path')
  parser.add_argument('--valid-num', type=int, default=0, help='number of validation data')

  args = parser.parse_args()
  train(args)


main()