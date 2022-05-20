# Oracle-MNIST

[![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](README.zh-CN.md)

`Oracle-MNIST` dataset comprises of 28×28 grayscale images of 30,222 ancient characters from 10 categories, for benchmarking pattern classification, with particular challenges on image noise and distortion. The training set totally consists of 27,222 images, and the test set contains 300 images per class. 

**1. Easy-of-use.** `Oracle-MNIST` shares the same data format with [the original MNIST dataset](http://yann.lecun.com/exdb/mnist/), allowing for direct compatibility with all existing classiﬁers and systems.

**2. Real-world challenge.** `Oracle-MNIST` constitutes a more challenging classification task than MNIST. The images of oracle characters suffer from 1) extremely serious and unique noises caused by three- thousand years of burial and aging and 2) dramatically variant writing styles by ancient Chinese, which all make them realistic for machine learning research. 

Oracle characters are the oldest hieroglyphs in China. Here's an example of how the data looks (*each class takes two-rows*):
<div align=center>
<img src="https://raw.githubusercontent.com/wm-bupt/images/main/oracle-mnist.png" width="800">
</div>

## Get the Data

You can directly download the dataset from [Google drive](https://drive.google.com/drive/folders/1JtckCILRwVloa54_DQA5zBTv4e5NJCgs?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1HXbr-23ib4aISOQKXy3HzQ) (code: 5pq5). The data is stored in the **same** format as the original MNIST data. The result files are listed in following table.

| Name  | Content | Examples | Size |
| --- | --- |--- | --- |
| `train-images-idx3-ubyte.gz`  | training set images  | 27,222|12.4 MBytes |
| `train-labels-idx1-ubyte.gz`  | training set labels  |27,222|13.7 KBytes |
| `t10k-images-idx3-ubyte.gz`  | test set images  | 3,000|1.4 MBytes |
| `t10k-labels-idx1-ubyte.gz`  | test set labels  | 3,000| 1.6 KBytes |

Alternatively, you can clone this GitHub repository; the dataset appears under `data/oracle`. This repo also contains some scripts for benchmark.

`Note`: All of the scanned images in Oracle-MNIST are preprocessed by the following conversion pipeline. We also make the original images available and left the data processing job to the algorithm developers. You can download the original images from [Google drive](https://drive.google.com/file/d/1gPYAOc9CTvrUQFCASW3oz30lGdKBivn5/view?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/15nPiaQ-HwcvfZx_o0qAaoQ) (code: 7aem).
<div align=center>
<img src="https://raw.githubusercontent.com/wm-bupt/images/main/convert.png" width="700">
</div>

## Usage

### Loading data with Python (requires [NumPy](http://www.numpy.org/))

Use `src/mnist_reader` in this repo:
```python
import mnist_reader
x_train, y_train = mnist_reader.load_data('data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('data/oracle', kind='t10k')
```

### Loading data with Tensorflow
Make sure you have [downloaded the data](#get-the-data) and placed it in `data/oracle`. Otherwise, *Tensorflow will download and use the original MNIST.*
```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/oracle')

data.train.next_batch(BATCH_SIZE)
```

`Note`:This official packages `tensorflow.examples.tutorials.mnist.input_data` would split training data into two subset: 22,222 samples are used for training, and 5,000 samples are left for validation. You can instead use `src/mnist_reader_tf` in this repo to load data. The number of validation data can be arbitrarily changed by varying the value of `valid_num`: 
```python
import mnist_reader_tf as mnist_reader
data = mnist_reader.read_data_sets('data/oracle', one_hot=True, valid_num=0)

data.train.next_batch(BATCH_SIZE)
```

## How to train it

You can reproduce the results of CNN by running `src/train_pytorch.py` or `src/train_tensorflow_keras.py`, and reproduce the results of other machine learning algorithms by running `benchmark/runner.py` provided by [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/benchmark).

CNN (pytorch)：
```bash
python train_pytorch.py --lr 0.1 --epochs 15 --net Net1 --data-dir ../data/oracle/
```

CNN (tensorflow+keras)：
```bash
python train_tensorflow_keras.py --lr 0.1 --epochs 15 --data-dir ../data/oracle/
```

## Citing Oracle-MNIST
If you use Oracle-MNIST in a scientific publication, we would appreciate references to the following paper:

**Oracle-MNIST: a Realistic Image Dataset for Benchmarking Machine Learning Algorithms. Mei Wang, Weihong Deng. [arXiv:2205.09442](https://arxiv.org/abs/2205.09442)**

Biblatex entry:
```latex
@article{wang2022oracle,
  title={Oracle-MNIST: a Realistic Image Dataset for Benchmarking Machine Learning Algorithms},
  author={Wang, Mei and Deng, Weihong},
  journal={arXiv preprint arXiv:2205.09442},
  pages={1--7},
  year={2022}
}
```
