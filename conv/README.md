# CUDA 卷积实现与优化
本文主要实现卷积的im2col优化算法，所有数据均基于NCHW格式。

### 实验环境

| 软硬件 | 版本/属性                              |
| ------ | -------------------------------------- |
| g++    | 5.4.0 20160609                         |
| cuda   | 10.1.168                               |
| 硬件   | TITAN X (Pascal)，理论浮点峰值11TFlops |
| OS     | ubuntu1~16.04.12                       |


### 1x1逐点卷积优化

1x1卷积可以直接转化为矩阵乘输入无需做im2col变换。这里测试了两种规格，相对于cudnn中卷积的加速比如下：

input=(64, 64, 64, 64),  kernel=(64, 64, 1, 1), speedup=0.90

input=(64, 128, 128, 128),  kernel=(128, 128, 1, 1), speedup=0.96

（为了编程方便，这里没有进行额外对齐操作，而是直接采用了对齐64的形状，不影响性能测量，但运行结果对其他形状不鲁棒）


### im2col卷积优化

这里实现了通用的卷积优化，支持不同stride, pad, dilation等功能。pytorch和mxnet的做法是逐样本进行im2col转换操作，然后再进行矩阵乘实现卷积。本文做法略有不同，主要体现在两个方面：一方面是逐样本进行转换操作会增加线程创建和销毁的开销，在内存充足的情况下可考虑直接将所有样本转换为一个大的column，然后做batch矩阵乘法；第二个方面是这里调用的是自己实现的gemm，需要进行对齐操作，因此对kernel和column进行对齐，乘完之后再进行一次数据拷贝，去除多余的零元素即可，其中column的对齐操作可以整合到im2col过程中，可避免对齐产生的额外数据拷贝。

这里对相对于cudnn的加速比如下：

input=(32, 64, 64, 64),  kernel=(64, 64, 3, 3), output=(32, 64, 62, 62), stride=1, pad=0, dilation=1, speedup=0.42

input=(32, 64, 64, 64),  kernel=(64, 64, 3, 3), output=(32, 64, 33, 33), stride=2, pad=2, dilation=1, speedup=0.69

input=(32, 128, 128, 128),  kernel=(128, 128, 3, 3), output=(32, 128, 126, 126), stride=1, pad=0, dilation=1, speedup=0.38

input=(32, 128, 128, 128),  kernel=(128, 128, 3, 3), output=(32, 128, 65, 65), stride=2, pad=2, dilation=1, speedup=1.75

input=(32, 128, 128, 128),  kernel=(128, 128, 4, 4), output=(32, 128, 65, 65), stride=2, pad=2, dilation=1, speedup=1.04

input=(32, 128, 128, 128),  kernel=(128, 128, 5, 5), output=(32, 128, 64, 64), stride=2, pad=2, dilation=1, speedup=0.71

为了验证在实践中的效果，这里以inceptionv3网络中使用到的前三个卷积层为例，数据如下：

| 卷积层名称       |  相对于cudnn加速比            |
| --------------- | ---------------------------- |
| Conv2d_1a_3x3   | 0.22                         |
| Conv2d_2a_3x3   | 0.39                         |
| Conv2d_2b_3x3 | 0.57 |
| Conv2d_3b_1x1 | 0.54 |

注：卷积层详细参数可参考[pytorch的inceptionV3实现](https://github.com/pytorch/vision/blob/12b551e7a7232d829df0f01ae9f6c56305571dfc/torchvision/models/inception.py)或者[tensorflow的inceptionV3实现](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)

### 总结

im2col卷积的核心是im2col的实现，以及矩阵乘法的优化。在学习im2col的过程中参考了pytorch和mxnet的思路，并且发现了pytorch的im2col算法的bug，去github最新分支进行核实时发现已经被改过来了，社区的力量果然强大。

另外尝试使用cublas的gemm来替代自己的gemm算法，但实际效果并不好，可能是因为cublas多次循环启动kernel开销较大，而自己的kernel已经把batch_size考虑进去了，没有多次启动kernel的开销。