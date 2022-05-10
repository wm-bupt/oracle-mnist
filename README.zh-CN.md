# Oracle-MNIST

[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](README.md)

`Oracle-MNIST` 数据集涵盖了来自10种类别的共30222个古文字的28×28的灰度图片，可以用它来测试机器学习算法在图像噪声和失真环境下的鲁棒性。训练集总共包含27222个图片，测试集每个类别包含300张图片。

Oracle-MNIST的格式与原始的[MNIST数据集](http://yann.lecun.com/exdb/mnist/)完全一致，因此，它与所有现有分类器和系统直接兼容，但却构成了比MNIST更具挑战性的分类任务。古文字的图片包含 **1）由近三千年的埋葬和老化所造成的极其严重的噪音和磨损，2）由古汉语书写风格造成的巨大类内变化**，从而能够为机器学习算法提供一个更加真实且更有难度的测试数据。

这个数据集的样子大致如下（每个类别占两行）：

<div align=center>
<img src="https://raw.githubusercontent.com/wm-bupt/images/main/oracle-mnist.png" width="800">
</div>

## 获取数据

你可以直接从[谷歌网盘](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)或者[百度网盘](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)下载该数据集。`Oracle-MNIST`的数据集的存储方式和命名与[经典MNIST数据集](http://yann.lecun.com/exdb/mnist/)完全一致。下表列出了相关的文件信息。

| 名称  | 描述 | 样本数量 | 文件大小 |
| --- | --- |--- | --- |
| `train-images-idx3-ubyte.gz`  | 训练集的图像  | 27,222|12.4 MBytes |
| `train-labels-idx1-ubyte.gz`  | 训练集的类别标签  |27,222|13.7 KBytes |
| `t10k-images-idx3-ubyte.gz`  | 测试集的图像  | 3,000|1.4 MBytes |
| `t10k-labels-idx1-ubyte.gz`  | 测试集的类别标签  | 3,000| 1.6 KBytes |

或者，你可以直接克隆这个代码库。数据集就放在`data/oracle`下。这个代码库还包含了一些用于评测的脚本。

`注意': `Oracle-MNIST`中所有的拓片甲骨文图片都经过以下步骤进行预处理。我们也将原始（未经过预处理）的拓片甲骨文图片开源，以供研究者们自己进行预处理工作。原始的图片可以从[谷歌网盘](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)或者[百度网盘](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)下载.

<div align=center>
<img src="https://raw.githubusercontent.com/wm-bupt/images/main/convert.png" width="700">
</div>

## 如何载入数据？

### 使用Python (需要安装`numpy`)
- 你可以直接使用`src/mnist_reader`：
```python
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/oracle', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/oracle', kind='t10k')
```

### 使用Tensorflow
请确保你已经[下载了我们的数据集](#获取数据)并把它放到了`data/oracle`下。不然， *Tensorflow会自动下载并使用原始的MNIST。*

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/oracle')

data.train.next_batch(BATCH_SIZE)
```

## 如何测评

你可以运行`src/main.py`对卷积神经网络的结果进行重现, 也可以通过运行[Fashion-MNIST网站](https://github.com/zalandoresearch/fashion-mnist/tree/master/benchmark)上提供的`benchmark/runner.py`对其他算法的结果进行重现.

卷积神经网络：
```bash
python main.py --gpu 0 --gen_img_dir generate_img/STSN --num_steps 250000 --batch_size 16
```

## 在论文中引用Oracle-MNIST
如果你在你的研究工作中使用了这个数据集，欢迎你引用这篇论文：

**Oracle-MNIST: a Realistic Image Dataset for Benchmarking Machine Learning Algorithms. Mei Wang, Weihong Deng. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747)**

亦可使用Biblatex:
```latex
@online{xiao2017/online,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Oracle-MNIST: a Realistic Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  eprintclass  = {cs.LG},
  eprinttype   = {arXiv},
  eprint       = {cs.LG/1708.07747},
}
```
