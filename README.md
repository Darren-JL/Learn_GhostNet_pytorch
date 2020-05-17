# Learning-GhostNet-Pytorch
This repository is created for learning the GhostNet paper and code, the code copy from "https://github.com/iamhankai/ghostnet.pytorch"



#### 一、论文理解

**一、理解原理**

摘要

（1）很多成功的神经网络都有一个重要特征--特征图中有冗余，但在神经结构设计中却鲜有研究；

（2）本文提出了一种新颖的Ghost模块，用较低的代价来生成更多的特征图。基于一系列固有的特征图，应用一系列线性变换，生成多个能充分反映特征信息的Ghost特征图；

（3）所提出的Ghost模块可以作为即插即用的组件对现有的卷积神经网络进行升级；

（4）Ghost模块堆叠成Ghost bottlenecks，从而建立轻量级的GhostNet；

（5）GhostNet在ImageNet数据集上的top-1精确度为75.7%，与MobileNetV3相比，耗费相似的代价，但是获得更好的效果。



1.简介

（1）深度神经网络设计的最新趋势是探索可移植、高效、性能可接受的移动设备网络架构，这些高效的深度网络可以为自动搜索方法提供新的搜索单元；

（2）在训练好的深度神经网络的特征图中，丰富甚至冗余的信息常常保证了对输入数据的全面理解，比如ResNet-50的特征图中，存在很多相似的特征图对，称为‘ghost’；

（3）这篇文章提出了一个新颖的Ghost模块，通过使用更少的参数来生成更多的特征。具体来说，一个普通的卷积层被拆成两部分。第一部分是普通的卷积，但是数量会受到严格控制，第二部分是使用一系列简单的线性算子来生成更多的特征图；

（4）在不改变输出特征图大小的情况下，与普通卷积神经网络相比，该Ghost模块减少了总体所需的参数数量和计算复杂度；

（5）首先在基准神经结构中替换原始的卷积层来证明Ghost模块的有效性，然后在多个基准视觉数据集上验证了GhostNets的优越性。实验结果表明，所提出的Ghost模块能够在保持相似识别性能的同时降低一般卷积层的计算成本，并且GhostNets能够在移动设备上快速推理的各种任务上超越MobileNetV3等最先进的高效深度模型。



2.相关工作

（1）模型压缩

（2）袖珍模型设计

MobileNets[17]是一系列基于深度可分卷积的轻量级深度神经网络。



3.方法介绍

3.1.Ghost模块

**原理推导：**

考虑到主流CNNs计算的中间特征图中广泛存在的冗余，文章提出减少所需资源，即生成所需资源的卷积滤波器。

卷积层生成特征图可以写成公式：
$$
Y=X*f+b
$$
其中$X\in R^{c\times h\times w}$是输入数据，$c$是输入数据的通道，$h,w$分别为输入数据的高和宽，如果产生$n$个特征图，则$Y\in R^{h^\prime\times w^\prime\times n}$，对于的卷积核$f\in R^{c\times k\times k\times n}$，这个卷积操作需要的FLOPs为$n\cdot h^\prime\cdot w^\prime\cdot c\cdot k\cdot k$，一般滤波器个数$n$和通道数$c$基本上都很大，所以导致一个卷积层的FLOPs比较大。
$$
Y=X*f+b，X\in R^{c\times h\times w}，Y\in R^{h^\prime\times w^\prime\times n}，f\in R^{c\times k\times k\times n}
$$


卷积层输出的特征图包含了非常多的冗余，当中有一些非常相似，所以没必要通过这么多FLOPs和参数的卷积操作去一个个的生成这些特征图，而是假设这些输出的特征图是一些少数固有特征图和他们的"伪影”(ghosts，通过很少的变换产生的)，这些固有特征图是由传统的卷积产生的。



假设输出的特征图中，有$m$个固有特征图，$Y^\prime \in R^{h^\prime\times w^\prime\times m}$是通过传统的卷积操作得到：
$$
Y^\prime=X*f^\prime
$$
其中$f^\prime \in R^{c\times k\times k\times m}$是传统运算的卷积核，$m<n$，为了简单起见，省略了偏执项，并且filter size, stride, padding等参数与之前的普通卷积保持一致。

为了进一步得到相同的$n$个特征图，利用下面的简单的线性映射在固有特征图$Y^\prime$ 上产生$s$个Ghost特征图:
$$
y_{ij}=\Phi_{i,j}(y_i^\prime),\forall i=1,...,m,j=1,...,s
$$
其中$y_i^\prime$是固有特征图$Y^\prime$中第$i$个特征图，$\Phi_{i,j}$是生成第$j$个特征图$y_{ij}$的线性算子，也就是说对于固有特征图$y^\prime_i$会有多个ghost特征图$\{y_{ij}\}^s_{j=1}$，且$\Phi_{i,s}$是恒等映射用来保留固有特征图（恒等映射和线性算子并行，保留了固有特征图）。通过上述这些线性映射可以得到最终的输出特征图$Y=[y_{11},y_{12},...,y_{ms}]$，通道数$n=m\cdot s$。

![ghost module](/Users/momo/Documents/momo学习笔记/工作汇报/ghost module.png)



**复杂度分析（模型有效的原因）：**

先补充点基础数学知识：

1.线性映射：是从一个线性空间$V$到另一个线性空间W的映射且保持和加法运算和数量乘法运算；

2.线性算子：是线性空间$V$到其自身的线性映射；

3.卷积运算是线性算子：
$$
kernel*(f_1+f_2)=kernel*f_1+kernel*f_2\\
kernel*(\alpha f)=\alpha\,\,kernel*f
$$
基于此，文章里面上述的作用在特征图上的线性算子$\Phi_{i,j}$都用二维平面卷积运算（不考虑通道）来代替。



分析使用Ghost模块在内存使用和理论上加速的好处。

对于$\Phi_{i,j},\forall i=1,...,m,j=1,...,s$，对于任何一个$i$，将第$i$个固有特征图映射到第$s$个特征图的线性映射$\Phi_{i,s}$是一个恒等映射，所以还剩下$m(s-1)=\frac{n}{s}\cdot (s-1)$个线性算子，假设每个线性算子（二维平面上的卷积运算）的卷积核的平均尺寸为$d\times d$，理想情况下是每个线性算子的卷积核尺寸都不一样。所以Ghost模块操作需要的FLOPs为$\frac{n}{s}\cdot h^\prime\cdot w^\prime\cdot c\cdot k\cdot k+(s-1)\cdot \frac{n}{s}\cdot h^\prime\cdot w^\prime\cdot d\cdot d$，与传统的卷积运算做加速比：
$$
\begin{align}
r_s&=\frac{n\cdot h^\prime\cdot w^\prime\cdot c\cdot k\cdot k}{\frac{n}{s}\cdot h^\prime\cdot w^\prime\cdot c\cdot k\cdot k+(s-1)\cdot \frac{n}{s}\cdot h^\prime\cdot w^\prime\cdot d\cdot d}\\
&=\frac{c\cdot k\cdot k}{\frac{1}{s}\cdot c\cdot k\cdot k+\frac{s-1}{s}\cdot d\cdot d}\\
&\approx\frac{s\cdot c}{s+c-1}\\
&\approx s
\end{align}
$$
此处得到的加速比有两次约等于，分别用到了$d\times d$和$k\times k$有相同的数量级，同时$s\ll c$.



**Ghost模块总结：Ghost模块用少量卷积核在多通道信号上的卷积得到一部分feature map，同时对少量卷积核的feature map做线性变换再得到feature map，然后两部分的feature map结合来代替多个卷积核在多通道信号上的卷积得到的feature map；**

**实际上文章中所谓的线性算子就是单通道的卷积运算，所以模型原理可以简单描述为：将多个卷积核在多通道上的卷积运算过程化简为少量卷积核在多通道上的卷积运算和单通道卷积运算**；



**二、论文模型结构**

3.2构建高效卷积神经网络

**Ghost Bottlenecks**

Ghost Bottlenecks由ghost模块构成，整体布局和ResNet中的Bottlenecks非常类似，ResNet-50中也有两种类型的Bottlenecks，一种是主路三个卷积层，然后加一个shortcut；另一种是主路三个卷积层，然后shortcut上再加一个卷积层。这里也提出了两种Ghost Bottlenecks。

第一种提出的Ghost Bottlenecks主要由两个堆叠的ghost模块组成。第一个ghost模块作为一个扩展层增加通道的数量。我们把输出通道数与输入通道数之比称为扩展比。第二个ghost模块减少通道的数量以匹配shortcut。然后将这两个ghost模块的输入和输出连接起来。

![Ghost Bottlenecks](/Users/momo/Documents/momo学习笔记/工作汇报/Ghost Bottlenecks.png)

第二种提出的Ghost Bottlenecks与第一种不一样的是，在中间加了一个stride为2的depthwise convolution，且shortcut这一条路是下采样的结果，并不是恒等映射。在所提出的两个Ghost Bottlenecks中，ghost模块中的基础卷积都是pointwise convolution，为了提高效率。

问题：

（1）何为下采样？（https://www.jianshu.com/p/fd9e2166cfcc）

i）上采样简单理解就是将图像放大，比如在图像分割当中，需要做像素级别的分类，因此在卷积提取特征后需要通过上采样将feature map还原到原图中。常见的上采样方法有双线性插值和转置卷积（反卷积）和上池化。

双线性插值就是两个方向的插值，两次线性插值；反卷积实际上也是一种卷积，只不过先在图像像素填充值，再进行卷积，总体来说图像还是扩大了；上池化就是直接在池化的像素周围填充0。

ii）下采样简单理解就是将图像缩小，最典型的就是池化操作。

（2）Depthwise卷积与Pointwise卷积（https://blog.csdn.net/tintinetmilou/article/details/81607721）

这些结构和常规卷积操作类似，可用来提取特征，但相比于常规卷积操作，其参数量和运算成本较低，所以在一些轻量级网络中会碰到这种结构，如MobileNet。

i）depthwise convolution

不同于常规卷积操作，Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积。上面所提到的常规卷积每个卷积核是同时操作输入图片的每个通道。Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map，**即只改变图像大小，不改变通道数。**

![depthwise convolution](/Users/momo/Documents/momo学习笔记/depthwise convolution.png)



ii）pointwise convolution

Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map，**即不改变大小，只改变通道数。**

![pointwise convolution](/Users/momo/Documents/momo学习笔记/pointwise convolution.png)

**Ghost Bottlenecks总结：联系后续的GhostNet结构发现，stride为1的Ghost Bottlenecks，在后续使用中输出和输入尺寸完全一样；stride为2的Ghost Bottlenecks，在后续使用中输出和输入尺寸减半；这是因为stride为1的Ghost Bottlenecks中基础卷积使用的是pointwise convolution，只改变通道数，不改变大小；stride为2的Ghost Bottlenecks，中间还隔着一层depthwise convolution，不改变通道数，但是会改变大小（减半），所以总体尺寸是减半的，因为前后两个ghost 模块中的卷积是pointwise convolution，不改变大小。**



**GhostNet**

因为MobileNetV3的优越性，Ghost的基本结构也遵循了它的设计，用我们的Ghost Bottlenecks代替了MobileNetV3中的Bottlenecks。GhostNet主要由一堆以ghost模块为构件的Ghost Bottlenacks组成。（如下图所示）

![GhostNet](/Users/momo/Documents/momo学习笔记/GhostNet.png)

为了定制所需的网络需求，我们可以简单地乘以一个系数$\alpha$来统一控制通道的数量，从而改变网络的宽度。

问题：

（1）MobileNet(V1,V2,V3)系列网络？ShuffleNet网络？

MobileNet系列和ShuffleNet网络是为移动设备设计的轻量级通用计算机视觉神经网络，实现分类/目标检测/语义分割多目标任务。

i)MobileNetV1

1）使用深度可分离卷积（depthwise separable convolutions）替代传统卷积，深度可分离卷积的过程与传统卷积的过程相比，计算量和参数都少很多，但是输入输出的尺寸不变。在第一阶段的depthwise convolution的运算过程中，比传统卷积共享了更多的参数和运算过程（https://blog.csdn.net/sinat_26114733/article/details/89076714）；
2）引入了两个收缩超参数（shrinking hyperparameters）：宽度乘子（width multiplier）和分辨率乘子（resolution multiplier）；



![MobileNetV1](/Users/momo/Documents/momo学习笔记/MobileNetV1.png)



ii)MobileNetV2（https://blog.csdn.net/u010712012/article/details/95922901）

1）在实际使用MobileNetV1的时候，发现深度卷积部分的卷积核比较容易训废掉：训完之后发现深度卷积训出来的卷积核有不少是空的，原因是有ReLU导致的信息损耗，解决方案是将ReLU替换成线性激活函数，即在MobileNetV1的基础上将pointwise convolution后面一层的ReLU6激活函数变成线性激活函数；

2）depthwise convolution本身没有改变通道的能力，来的是多少通道输出就是多少通道。如果来的通道很少的话，DW深度卷积只能在低维度上工作，这样效果并不会很好，所以文章“扩张”通道。既然我们已经知道PW逐点卷积也就是1×1卷积可以用来升维和降维，那就可以在DW深度卷积之前使用PW卷积进行升维（升维倍数为t，t=6），再在一个更高维的空间中进行卷积操作来提取特征；

3）回顾V1的网络结构，发现V1很像是一个直筒型的VGG网络。我们想像Resnet一样复用我们的特征，所以我们引入了shortcut结构；

综合以上三点，可以对比下MobileV1,V2:

![MobileNetV1V2](/Users/momo/Documents/momo学习笔记/MobileNetV1V2.png)

以及MobileV2,ResNet：

![MobileNetV2ResNet](/Users/momo/Documents/momo学习笔记/MobileNetV2ResNet.png)

可以发现，都采用了 1×1 -> 3 ×3 -> 1 × 1 的模式，以及都使用Shortcut结构。但是不同点：
ResNet 先降维 (0.25倍)、卷积、再升维。
MobileNetV2 则是 先升维 (6倍)、卷积、再降维。
刚好V2的block刚好与Resnet的block相反，作者将其命名为Inverted residuals。就是论文名中的Inverted residuals。



iii)MobileNetV3

![V2Bottlenecks](/Users/momo/Documents/momo学习笔记/V2Bottlenecks.png)



![V3Bottlenecks](/Users/momo/Documents/momo学习笔记/V3Bottlenecks.png)

上面两张图是MobileNetV2和MobileNetV3的网络块结构。可以看出，MobileNetV3是综合了以下三种模型的思想：MobileNetV1的深度可分离卷积（depthwise separable convolutions）、MobileNetV2的具有线性瓶颈的逆残差结构(the inverted residual with linear bottleneck)和MnasNet的基于squeeze and excitation结构的轻量级注意力模型。

除此之外在MobileNetV2的基础上将最后一步的平均池化层前移并移除最后一个卷积层，引入h-swish激活函数。

![V3laststage](/Users/momo/Documents/momo学习笔记/V3laststage.png)



下图是MobileNetV3的结构：

![MobileNetV3](/Users/momo/Documents/momo学习笔记/MobileNetV3.png)

可以看出文章中提出的GhostNet模型结构用Ghost Bottlenecks代替了MobileNetV3中的Bottlenecks，剩下的模型中不同层的输入输出规模与MobileNetV3的基本一样。



（2）The squeeze and excite (SE) module？GhostNet使用SE module的地方如何理解？是否像MobileNetV3一样在Bottlenecks中添加了SE module?



（3）列表中的exp(expansion size)指什么?



**总结：ResNet,MobileNetV1,V2,V3,GhostNet的区别和相似点**

**1.ResNet和MobileNetV1是最基本的两个trick，一个提出残差结构，引入shortcut；另一个将传统的卷积分离。后续的MobileNetV2,V3和GhostNet都采用了残差结构和卷积分离；**

**2.MobileNetV2在MobileNetV1的block之前添加了一层pointwise convolution来增加通道数，且添加了了shortcut，和ResNet相比形状相反，相当于Inverted residuals；**

**3.MobileNetV3是综合了以下三种模型的思想：MobileNetV1的深度可分离卷积（depthwise separable convolutions）、MobileNetV2的具有线性瓶颈的逆残差结构(the inverted residual with linear bottleneck)和MnasNet的基于squeeze and excitation结构的轻量级注意力模型；**

**4.GhostNet在MobileNetV3的基础上，将Bottlenecks替换成Ghost Bottlenecks，Ghost Bottlenecks本身也是一个残差结构，其中的ghost module也是传统卷积的另一种分解和加速。**



**三、复现论文精度**

4.实验

在本节中，我们首先用所提出的Ghost模块替换原有的convolutional layers来验证其有效性。然后，使用新模块构建的GhostNet架构将在图像分类和目标检测基准上进行进一步测试。

（1）Toy Experiments

这个实验实际上是在验证线性变换的形式，对于ResNet-50中成对的feature maps，探索了其他一些低成本的线性操作来构造Ghost模块，如仿射变换和小波变换，但是效果都不如单通道的卷积，验证了不同尺寸的卷积核卷积前后的误差，确定尺寸$d$的大小。

![experiment1](/Users/momo/Documents/momo学习笔记/experiment1.png)

（2）Ghost module on CIFAR-10

在VGG-16和ResNet-56上，用ghost module去代替普通的卷积，来来验证ghost module的有效性，同时验证超参数$d,s$的取值。

![experiment21](/Users/momo/Documents/momo学习笔记/experiment21.png)

![experiment22](/Users/momo/Documents/momo学习笔记/experiment22.png)

（3）Ghost module on ImageNet

用ghost module模块去代替ResNet-50中传统的卷积层，在ImageNet上做分类实验。

![experiment3](/Users/momo/Documents/momo学习笔记/experiment3.png)



（4）GhostNet on ImageNet

在ImageNet数据集上验证GhostNet，可以看出相同运算次数情况下，GhostNet比其他网络精度高；相同精度下，GhostNet所需的计算次数比其他网络少。

![experiment4](/Users/momo/Documents/momo学习笔记/experiment4.png)



（5）GhostNet on MS COCO dataset

GhostNet在COCO数据集上的表现最好。

![experiment5](/Users/momo/Documents/momo学习笔记/experiment5.png)



#### 二、代码复现

对于pytorch版本，官方的repository中只给出了一个.py文件ghost_net.py，并没有训练和测试代码。这里学习和解读ghost_net.py代码文件重要部分，同时在ImageNet上进行训练和测试。

##### 1.SE module

先简单介绍SE module:（https://www.cnblogs.com/Libo-Master/p/9663508.html），再看代码。

**SE module 结构**：

（1）$F_{tr}$：$X$到$U$的卷积过程 ，但是通道之间的关系并没有发生变化；

（2）$F_{sq}$：将每个通道做了一个$squeeze$操作（主要是池化操作），将每个通道表示成了一个标量，得到per channel的描述；

（3）$F_{ex}$：将per channel标量进行“激活”，可以理解为算出了per channel的$W$（权值），实际上这一步就是全连接；

（4）最后将per channel的$W$（权重）乘回到原来的feature map上得到加权后的channel，将channel 做了恰当的融合；

SE-Module 可以用于网络的任意阶段，且**squeeze 操作保证了在网络的早期感受野就可以大到全图的范围。**

![SE module](/Users/momo/Documents/momo学习笔记/SE module.png)



**具体实践中的SE module有SE-inception Module and SE-ResNet Module:**

左图是SE-inception Module，第（2）步中的$squeeze$采用average pooling，得到$1*1*C$的向量；后面再接FC，但是为了减少参数，做了降维操作，增加了一个降维系数$r$，输出$1*1*\frac{C}{r}$；后接$ReLU$，再做一个升维操作，得到$1*1*C$，最终采用$Sigmoid$函数激活。激活之后，将每一个通道的权值向量$1*1*C$乘到相应的通道上，结构如下：

![inception resnet SE module](/Users/momo/Documents/momo学习笔记/inception resnet SE module.png)

可以看到参数量主要取决与FC，在实验时$r$一般取16，经验值！右图中，是resnet module，改造和inception分支很类似。

**代码段：**

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y
```

（1）nn.AdaptiveAvgPool2D(outputsize)（https://www.zhihu.com/question/282046628/answer/426193495）

二元自适应均值汇聚层，就是将输入的二维数据（二元）进行下采样操作（汇聚层），操作方式是均值池化（均值），步长和池化核大小根据输出的尺寸（outputsize）自己适应（自适应）。

（2）x.view(b,c)

将x变换成形状为b*c的tensor。

（3）torch.clamp(y,0,1)

限制y（权重）的取值在[0,1]之间。

上述的SE module实际上就是SE-inception Module。



##### 2.depthwise convolution

```python
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )
```

（1）nn.Sequential()

按照顺序将各个过程添加到计算图当中。

（2）class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

groups参数的含义：其意思是将对应的输入通道与输出通道数进行分组, 默认值为1, 也就是说默认输出输入的所有通道各为一组。 比如输入数据大小为90x100x100x32，通道数32，要经过一个3x3x48的卷积，group默认是1，就是全通道一起进行卷积计算。

如果group是2，那么对应要将输入的32个通道分成2个16的通道，将输出的48个通道分成2个24的通道。对输出的2个24的通道，第一个24通道与输入的第一个16通道进行全卷积，第二个24通道与输入的第二个16通道进行全卷积。

极端情况下，输入输出通道数相同，group大小为32，那么每个卷积核的channel，只与输入的对应的通道进行卷积。



##### 3.Ghost Module

```python
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # 固有feature maps
        new_channels = init_channels*(ratio-1)  # 经过变换得到的feature maps

        # ghost module中的基础卷积，实际上是pointwise convolution，从参数kernel_size=1看出
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        #ghost module中的线性变换部分，实际上是depthwise convolution，从参数groups=i nit_channels看出
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
```



##### 4.Ghost Bottlenecks

```python
class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite，SE module放在Ghost Bottlenecks的中间
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        # 如果采用步长为1的Ghost Bottlenecks，shortcut就是恒等映射；如果步长为2的Ghost Bottlenecks，shortcut采用下采样，下采样是卷积的过程，先经过步长为2的depthwise convolution，再经过pointwise convolution，总体尺寸减半
        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
```



#### 5.GhostNet

```python
class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs  # 每一个Ghost Bottlenecks的卷积核大小，exp，输出channel数，是否使用SE module，以及步长

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)  # 确保所有层的输出channel数都是8的倍数
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            # exp_size是Ghost Bottlenecks中间第一个ghost module扩大channels的倍数
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            # layers中包含了所有层
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers，先卷积再池化，输出1*1*960
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),  # 最后一层全连接
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 所有的Ghost Bottlenecks
        x = self.squeeze(x)  # 最后层的卷积和池化
        x = x.view(x.size(0), -1)  # 将池化结果展开为960
        x = self.classifier(x)  # 最后的960-1280-1000全连接
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
def ghost_net(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)
```



上述就是文件ghost_net.py的结构，下面使用GhostNet进行ImageNet的分类训练和测试。

##### 6.GhostNet on ImageNet

network_ghost.py：

```python
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
import sys
sys.path.append(os.getcwd)
from ghost_net import ghost_net

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定一块gpu为可见
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定四块gpu为可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# #############创建数据加载器###################
print('data loaded begin!')
# 预处理，将各种预处理组合在一起
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

train_dataset = torchvision.datasets.ImageFolder(root='/mnt/data2/raw/train', transform=data_transform)
# 使用ImageFolder需要数据集存储的形式：每个文件夹存储一类图像
# ImageFolder第一个参数root : 在指定的root路径下面寻找图片
# 第二个参数transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
train_data = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
# 第一个参数train_dataset是上面自定义的数据形式
# 最后一个参数是线程数，>=1即可多线程预读数据

test_dataset = torchvision.datasets.ImageFolder(root='/mnt/data2/raw/validation', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)

print(type(train_data))
print('data loaded done!')
# <class 'torch.utils.data.dataloader.DataLoader'>


# ##################创建网络模型###################


# ImageNet预训练模块，更新所有层的参数
print('GhostNet model loaded begin!')
# 使用轻量级GhostNet,进行预训练
model = ghost_net(width_mult=1.0)
print(model)
print('GhostNet model loaded done!')
# 对于模型的每个权重，使其进行反向传播，即不固定参数
#for param in model.parameters():
#   param.requires_grad = True



# 修改最后一层的分类数
#class_num = 1000
#channel_in = model.fc.in_features  # 获取fc层的输入通道数
#model.fc = nn.Linear(channel_in, class_num)  # 最后一层替换


# ##############训练#################

# 在可见的gpu中，指定第一块卡训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), 1e-1)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-2)

nums_epoch = 20  # 训练epoch的数量，动态调整

print('training begin!')
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
S = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    model = model.train()
    print('Epoch ' + str(epoch+1) + ' begin!')
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)

        # 前向传播
        out = model(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        print('Train loss in current Epoch' + str(epoch+1) + ':' + str(loss))
        #print('BP begin!')
        # 反向传播
        loss.backward()
        #print('BP done!')
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        print('Train accuracy in current Epoch' + str(epoch+1) + ':' + str(acc))

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    print('Epoch' + str(epoch+1)  + ' Train  done!')
    print('Epoch' + str(epoch+1)  + ' Test  begin!')
    # 每个epoch测一次acc和loss
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img1, label1 in test_data:
        img1 = img1.to(device)
        label1 = label1.to(device)
        out = model(img1)

        loss = criterion(out, label1)
        # print('Test loss in current Epoch:' + str(loss))

        # 记录误差
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label1).sum().item()
        acc = num_correct / img1.shape[0]
        eval_acc += acc

    print('Epoch' + str(epoch+1)  + ' Test  done!')
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('Epoch {} ,Train Loss: {} ,Train  Accuracy: {} ,Test Loss: {} ,Test Accuracy: {}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
            eval_acc / len(test_data)))
    s = 'Epoch {} ,Train Loss: {} ,Train  Accuracy: {} ,Test Loss: {} ,Test Accuracy: {}'.format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data))
    S.append(s);

    torch.save(model, '/home/momo/sun.zheng/GhostNet_pytorch/ghost.pytorch/model_l_0.01_SGD_epoch_20.pkl')
    print('model saved done!')
    print(losses)
    print(acces)
    print(eval_losses)
    print(eval_acces)
    print(S)
```



这里的代码正常运行，但并没有使用GPU加速，下面采用pytorch官方github上给出的多机多卡训练代码和另外的一个大神写的三个单机多卡训练的代码。

pytorch官方给出的ImageNet上的训练代码包括多机多卡训练，情况非常复杂。（https://github.com/pytorch/examples/blob/master/imagenet/main.py）这里先介绍GitHub上另一个大神的代码，编写了三种情况下的单机多卡训练代码。（https://github.com/tczhangzhi/pytorch-distributed）

1.使用nn.DataParallel

2.使用torch.multiprocessing

3.使用torch.distributed



在这三种情况都介绍完之后，再回过头来介绍pytorch官方给出的最复杂的代码。

1.使用nn.DataParallel，dataparallel.py

DataParallel 使用起来非常方便，但速度是最慢的。我们只需要用 DataParallel 包装模型，再设置一些参数即可。需要定义的参数包括：参与训练的 GPU 有哪些，device_ids=gpus；用于汇总梯度的 GPU 是哪个，output_device=gpus[0] 。DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总，关键代码如下：

```python
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
```

```python
import csv

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/data2/raw', help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=3200,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    gpus = [0, 1, 2, 3]  # 采用四张显卡计算
    main_worker(gpus=gpus, args=args)


def main_worker(gpus, args):
    global best_acc1

    # create model，这一段相应替换成其他模型即可
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    log_csv = "dataparallel.csv"

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_end = time.time()

        with open(log_csv, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
            csv_write.writerow(data_row)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
```



2.使用torch.multiprocessing，multiprocessing_distributed.py

使用时，只需要调用 torch.multiprocessing.spawn，torch.multiprocessing 就会帮助我们自动创建进程。如下面的代码所示，spawn 开启了 nprocs=4 个线程，每个线程执行 main_worker 并向其中传入 local_rank（当前进程 index）和 args（即 4 和 myargs）作为参数：

```python
import torch.multiprocessing as mp
mp.spawn(main_worker, nprocs=4, args=(4, myargs))
```

在main_worker中有：

```python
def main_worker(proc, ngpus_per_node, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
   torch.cuda.set_device(args.local_rank)
   ......
```

完整代码为：

```python
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import csv

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/data2/raw', help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
    # create model，可以加载其他模型
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, gpu, args)
        return

    log_csv = "multiprocessing_distributed.csv"

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, gpu, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, gpu, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_end = time.time()

        with open(log_csv, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
            csv_write.writerow(data_row)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
```



3.使用 torch.distributed 加速并行训练

这个工具我不太清楚原理，因为涉及到参数--local_rank（'node rank for distributed training'），后面pytorch官网上的代码也有此参数，这应该是多机（多节点）训练时需要的，此处用在单机多卡上。

```python
import csv

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/zhangzhi/Data/ImageNet2012', help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=3200,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, 4, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    dist.init_process_group(backend='nccl')
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, gpu, args)
        return

    log_csv = "distributed.csv"

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, gpu, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, gpu, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_end = time.time()

        with open(log_csv, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
            csv_write.writerow(data_row)
            
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

```



总结一下三个工具，第一个使用nn.DataParallel，是最简单也是速度最慢的一个，最好理解原理；第二个使用torch.multiprocessing，个人理解是使用多线程，每个线程使用一个GPU，达到使用多GPU的目的；第三个使用torch.distributed，个人理解是使用多进程，节点（node）以及节点排序（node rank）。

三个工具都可以用于单机多卡训练，pytorch官网上的代码是同时使用了后两个工具，可以用于多机多卡训练，适当设置参数可以达到单机多卡的目的，个人不太清楚参数--local-rank（节点排序）的含义，所以没有成功执行过，下面给出个人注释的代码：

```python
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
            if name.islower() and not name.startswith("__")
                and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                            help='path to dataset')  # data没有--，是必填的参数
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')  # 模型保存路径
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')  # 节点数，并行的机器数，这里赋值为1
parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')  # 使用哪个GPU
parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')  # 是否进行多进程分布式训练，默认是True，但是如果不赋值，就是false，即在python运行时，python xx.py --multiprocessing-distributed，此时为True，如果直接pythonxx.py，此时为False
best_acc1 = 0
#y = parser.parse_args()
#print(y)
#print(y.lr)

def main():
        args = parser.parse_args()

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')  # 选择了一个特定的GPU，这回禁用数据并行性

        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

        args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # 为了使用单机多GPU训练，args.multiprocessing_distributedy应为True，args.world_size为1，所以args.distributed为True

        ngpus_per_node = torch.cuda.device_count()  #GPU数量，当前机器上为4块
        if args.multiprocessing_distributed:  # 如果分布式计算
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size  # 4卡乘以1机器，为4
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))  # 并发运行
        else:  # 不进行分布式计算运行函数main_worker()
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)

# mp.pawn 开启了 nprocs=4 个线程，每个线程执行main_worker并向其中传入local_rank（当前进程index）和 args（即 4 和 args）作为参数：
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu  # 此时的gpu是当前进程数，刚好对应0,1,2,3张卡，每个进程一张卡

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
```



采用multiprocessing_distributed.py训练：

1.参数设置如下：

初始学习率：0.4；

学习率衰减：每过30个epoch，学习率减为原来的0.1倍；

Weight_decay：1e-4；

迭代器：SGD，momemtum=0.9；

batch_size为512；



训练结果为：

......

Epoch 6:    acc@1:    32.404,    acc@5:    58.068

......

Epoch14:    acc@1:  34.292,    acc@5:    60.548

......

Epoch19：acc@1:  36.942,    acc@5:    63.170

只训练到20个epoch，考虑到训练结果中，到5个epoch的时候精度已经上升不多了，所以在5的倍数epoch的时候学习率减半继续训练。



2.更改参数如下，完全按照论文中的参数设置（除了线性学习率衰减不清楚具体参数设置）：

初始学习率：0.4；

学习率衰减：每过5个epoch，学习率减为原来的0.5倍；

Weight_decay：4e-5；

迭代器：SGD，momemtum=0.9；

batch_size为1024；



训练结果为：

......

epoch 0:    acc@1:    15.622%,    acc@5:    35.586%

......

epoch 2:    acc@1:    36.702,%    acc@5:    63.030%

epoch 3:    acc@1:    36.582%,    acc@5:    62.524%

epoch4：acc@1:    44.482%,    acc@5:    70.126%

......

epoch8：acc@1:    52.884%,    acc@5:    77.388%

......

epoch10：acc@1:    57.902%,    acc@5:    81.022%

epoch11：acc@1:    58.256%,    acc@5:    80.988%

epoch12：acc@1:    58.156%,    acc@5:    81.152%

......

epoch16：acc@1:    61.882%,    acc@5:    83.848%

epoch17：acc@1:    62.080%,    acc@5:    83.900%

epoch18：acc@1:    62.000%,    acc@5:    83.764%

epoch19：acc@1:    61.396%,    acc@5:    83.426%

......

epoch24：acc@1:    64.662%,    acc@5:    85.554%

epoch25：acc@1:    65.496%,    acc@5:    86.092%

epoch26：acc@1:    65.500%,    acc@5:    86.078%

epoch27：acc@1:    65.890%,    acc@5:    86.33%

......

epoch32：acc@1:    66.440%,    acc@5:    86.55%

......

epoch34：acc@1:    66.514%,    acc@5:    86.638%

epoch35：acc@1:    66.804%,    acc@5:    86.844%

......

epoch40：acc@1:    67.010%,    acc@5:    86.918%

epoch41：acc@1:    66.988%,    acc@5:    86.982%

......

epoch43：acc@1:    67.018%,    acc@5:    87.006%

......

epoch48：acc@1:    67.088%,    acc@5:    87.066%

epoch49：acc@1:    67.094%,    acc@5:    87.094%

epoch50：acc@1:    67.208%,    acc@5:    87.094%

epoch51：acc@1:    67.084%,    acc@5:    87.034%

epoch52：acc@1:    67.222%,    acc@5:    87.056%

......

epoch55：acc@1:    67.102%,    acc@5:    87.112%

......

epoch57：acc@1:    67.232%,    acc@5:    87.068%

epoch58：acc@1:    67.212%,    acc@5:    87.076%

epoch59：acc@1:    67.160%,    acc@5:    87.078%

......

epoch63：acc@1:    67.128%,    acc@5:    87.074%

epoch64：acc@1:    67.178%,    acc@5:    87.142%

......

epoch66：acc@1:    67.132%,    acc@5:    87.098%

epoch67：acc@1:    67.244%,    acc@5:    87.144%

......

epoch71：acc@1:    67.196%,    acc@5:    87.156%

......

停止训练！

保存此时的模型参数，将训练方式改为微调，学习率从0.05（适当提高学习率）开始线性递减到0，再训练30个epochs。加载方式如下：

（1）只保存模型参数以及加载：

```python
torch.save(model.state_dict(), PATH)  # 模型保存，一般保存为model.pth

model.load_state_dict(torch.load(PATH))  # 模型加载
```

（2）保存模型的参数和各个超参数为字典：

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)  # 保存模型为一个字典，一般保存为model.pth.tar

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])  # 模型加载
```



训练结果如下：

epoch0：acc@1:    65.258%,    acc@5:    86.848%

......

epoch4：acc@1:    68.668%,    acc@5:    88.730%

epoch5：acc@1:    69.520%,    acc@5:    89.180%

epoch6：acc@1:    69.560%,    acc@5:    89.288%

epoch7：acc@1:    69.112%,    acc@5:    88.790%

epoch8：acc@1:    68.602%,    acc@5:    88.680%

......

epoch13：acc@1:    70.522%,    acc@5:    89.756%

......

epoch16：acc@1:    71.066%,    acc@5:    90.014%

epoch17：acc@1:    71.142%,    acc@5:    90.192%

......

epoch20：acc@1:    71.944%,    acc@5:    90.544%

epoch21：acc@1:    72.108%,    acc@5:    90.556%

epoch22：acc@1:    72.228%,    acc@5:    90.556%

epoch23：acc@1:    71.999%,    acc@5:    90.456%

......

epoch27：acc@1:    72.924%,    acc@5:    91.056%

epoch28：acc@1:    73.092%,    acc@5:    90.992%

epoch29：acc@1:    73.200%,    acc@5:    91.162%

模型保存为model_best.pth.tar。



3.第三次学习率衰减方式调整

第二次调整的学习率衰减方式中top-1精度最高为67.2%，并且到了40个epoch之后基本上就没有提升的空间了。在github上找到别人的复现版本（https://github.com/d-li14/ghostnet.pytorch） ，调整学习率为线性衰减方式：

```python
lr = init_lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
```

省略warm-up的步骤，使用如下衰减公式：

```python
lr = init_lr * (1 - current_iter / max_iter)
```

最大迭代次数设置为240，学习率从0.4线性衰减到0，其余参数保持不变，训练结果如下：

epoch0：acc@1:    14.130%,    acc@5:    32.216%

epoch1：acc@1:    30.340%,    acc@5:    56.084%

epoch2：acc@1:    36.188%,    acc@5:    62.150%

epoch3：acc@1:    41.452%,    acc@5:    67.206%

......

epoch8：acc@1:    49.282%,    acc@5:    74.348%

epoch9：acc@1:    48.572%,    acc@5:    74.038%

epoch10：acc@1:    49.662%,    acc@5:    74.918%

epoch11：acc@1:    50.028%,    acc@5:    75.260%

epoch12：acc@1:    50.836%,    acc@5:    75.510%

......

epoch18：acc@1:    52.262%,    acc@5:    79.904%

epoch19：acc@1:    52.780%,    acc@5:    77.764%

epoch20：acc@1:    53.560%,    acc@5:    78.316%

......

epoch25：acc@1:    53.500%,    acc@5:    78.456%

epoch26：acc@1:    52.720%,    acc@5:    77.736%

epoch27：acc@1:    49.056%,    acc@5:    74.610%

epoch28：acc@1:    54.584%,    acc@5:    78.840%

......

epoch34：acc@1:    55.604%,    acc@5:    79.552%

......

epoch36：acc@1:    54.768%,    acc@5:    79.122%

epoch37：acc@1:    53.608%,    acc@5:    78.120%

......

epoch41：acc@1:    54.002%,    acc@5:    78.724%

epoch42：acc@1:    50.578%,    acc@5:    75.700%

......

epoch44：acc@1:    54.992%,    acc@5:    79.582%

epoch45：acc@1:    56.034%,    acc@5:    80.428%

......

epoch49：acc@1:    53.222%,    acc@5:    77.800%

......

epoch52：acc@1:    57.156%,    acc@5:    80.944%

epoch53：acc@1:    55.906%,    acc@5:    80.094%

......

epoch57：acc@1:    55.084%,    acc@5:    79.394%

......

epoch59：acc@1:    56.612%,    acc@5:    80.636%

epoch60：acc@1:    55.792%,    acc@5:    80.098%

......

epoch64：acc@1:    53.050%,    acc@5:    77.578%

epoch65：acc@1:    57.818%,    acc@5:    81.122%

epoch66：acc@1:    56.568%,    acc@5:    80.616%

......

epoch68：acc@1:    57.672%,    acc@5:    81.342%

......

epoch72：acc@1:    58.532%,    acc@5:    82.016%





**模型评测：**

首先注意数据的预处理方式，代码如下：

```
# Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)
```

数据加载预处理阶段可以分为两个阶段：

（1）torchvision.datasets.ImageFolder函数，这个函数负责数据的预处理，参数包括数据的目录，以及预处理transforms.Compose；其中数据的目录部分要求数据集是分门别类地存储在各个类别文件夹中，这样才可以返回各个样本的标签；

（2）torch.utils.data.DataLoader函数，这个函数是负责将（1）中处理好的数据按照batch_size进行打包；

一般使用pytorch的时候，数据加载和预处理部分都需要这两阶段。但是步骤（1）一般都不用torchvision.datasets.ImageFolder函数，而是自己写脚本来预处理特定的数据，步骤（2）较简单，直接使用pytorch自带的DataLoader函数即可。

下面给出评测该模型的脚本：

```python
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from ghost_net import ghost_net
import time

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形>转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])



# 加载模型
print('load model begin!')
model = ghost_net(width_mult=1.0)
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')



# 测试单张图像
img = Image.open('/home/sz/model_eval/panda.jpg')
img = data_transform(img)
#img = torch.Tensor(1,3,224,224) #如果只是测试时间，直接初始化一个Tensor即可
print(type(img))
print(img.shape)
#img_f = torch.Tensor(1, 3, 224, 224)
img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
print(type(img))
print(img.shape)

time_start = time.time()
img_= img.to(device)
outputs = model(img_)
time_end = time.time()
time_c = time_end - time_start
_, predicted = torch.max(outputs,1)
print('this picture maybe:' + str(predicted))
print('time cost:', time_c, 's')

'''
# 批量测试验证集中的图像，使用dataloader，可以更改batch_size调节测试速度
print('Test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
print(type(test_data))
print('Test data load done!')

torch.no_grad()
for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    print(pred)
    print(label1)
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
'''
```

评测的时候，先定义好transform，然后直接读取的图片进行预处理即可输入模型运算：

```python
img = data_transform(img)
```



以上是一个图像分类模型的例子，包括各个步骤（数据下载，网络模型代码编写，训练，以及评测）。文件夹evaluation中比较了当前工业界使用的一些轻量级网络的比较和评测。

