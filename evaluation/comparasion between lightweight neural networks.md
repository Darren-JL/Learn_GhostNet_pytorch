#### 验证GhostNet在移动端设备上使用的可行性

具体内容：将自己复现的GhostNet模型和已有的MobileNet系列，EfficientNet系列，MnasNet，ShuffleNet系列等轻量级网络做比较，主要比较一下几个方面：

1.比较网络结构；

2.比较模型的大小、papameter参数数量，以及FLOPs运算次数；

3.在ImageNet数据集上，比较模型的性能（精确度和运行时间）；



##### 一、网络结构对比

**1.GhostNet**

**(1)ghost module**

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

![ghost module](/Users/momo/Documents/momo学习笔记/ghost module.png)



**(2)Ghost Bottlenecks**

Ghost Bottlenecks由ghost模块构成，整体布局和ResNet中的Bottlenecks非常类似，ResNet-50中也有两种类型的Bottlenecks，一种是主路三个卷积层，然后加一个shortcut；另一种是主路三个卷积层，然后shortcut上再加一个卷积层。这里也提出了两种Ghost Bottlenecks。

第一种提出的Ghost Bottlenecks主要由两个堆叠的ghost模块组成。第一个ghost模块作为一个扩展层增加通道的数量。我们把输出通道数与输入通道数之比称为扩展比。第二个ghost模块减少通道的数量以匹配shortcut。然后将这两个ghost模块的输入和输出连接起来。

![Ghost Bottlenecks](/Users/momo/Documents/momo学习笔记/Ghost Bottlenecks.png)

第二种提出的Ghost Bottlenecks与第一种不一样的是，在中间加了一个stride为2的depthwise convolution，且shortcut这一条路是下采样的结果，并不是恒等映射。在所提出的两个Ghost Bottlenecks中，ghost模块中的基础卷积都是pointwise convolution，为了提高效率。



**2.MobileNetV1**

MobileNetV1将传统卷积的过程改进为深度可分离卷积，相比于常规卷积操作，其参数量和运算成本较低，具体分为两个阶段：

（1）depthwise convolution

不同于常规卷积操作，Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积。上面所提到的常规卷积每个卷积核是同时操作输入图片的每个通道。Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map，**即只改变图像大小，不改变通道数。**

![depthwise convolution](/Users/momo/Documents/momo学习笔记/depthwise convolution.png)



（2）pointwise convolution

Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map，**即不改变大小，只改变通道数。**

![pointwise convolution](/Users/momo/Documents/momo学习笔记/pointwise convolution.png)

**Ghost Bottlenecks总结：联系后续的GhostNet结构发现，stride为1的Ghost Bottlenecks，在后续使用中输出和输入尺寸完全一样；stride为2的Ghost Bottlenecks，在后续使用中输出和输入尺寸减半；这是因为stride为1的Ghost Bottlenecks中基础卷积使用的是pointwise convolution，只改变通道数，不改变大小；stride为2的Ghost Bottlenecks，中间还隔着一层depthwise convolution，不改变通道数，但是会改变大小（减半），所以总体尺寸是减半的，因为前后两个ghost 模块中的卷积是pointwise convolution，不改变大小。**



**3.MobileNetV2**

1）depthwise convolution本身没有改变通道的能力，来的是多少通道输出就是多少通道。如果来的通道很少的话，DW深度卷积只能在低维度上工作，这样效果并不会很好，所以MobileNetV2“扩张”通道。PW逐点卷积也就是1×1卷积可以用来升维和降维，那就可以在DW深度卷积之前使用PW卷积进行升维（升维倍数为t，t=6），再在一个更高维的空间中进行卷积操作来提取特征；

2）回顾MobileNetV1的网络结构，V1像是一个直筒型的VGG网络。MobileNetV2引入了shortcut结构；

可以对比下MobileV1,V2:

![MobileNetV1V2](/Users/momo/Documents/momo学习笔记/MobileNetV1V2.png)

以及MobileV2,ResNet：

![MobileNetV2ResNet](/Users/momo/Documents/momo学习笔记/MobileNetV2ResNet.png)

可以发现，都采用了 1×1 -> 3 ×3 -> 1 × 1 的模式，以及都使用Shortcut结构。但是不同点：
ResNet 先降维 (0.25倍)、卷积、再升维。
MobileNetV2 则是 先升维 (6倍)、卷积、再降维。
刚好MobileNetV2的block刚好与Resnet的block相反，文章作者将其命名为Inverted residuals。



**4.MobileNetV3**

MobileNetV2和MobileNetV3的结构对比：

![V2Bottlenecks](/Users/momo/Documents/momo学习笔记/V2Bottlenecks.png)



![V3Bottlenecks](/Users/momo/Documents/momo学习笔记/V3Bottlenecks.png)

上面两张图是MobileNetV2和MobileNetV3的网络块结构。可以看出，MobileNetV3是综合了以下三种模型的思想：MobileNetV1的深度可分离卷积（depthwise separable convolutions）、MobileNetV2的具有线性瓶颈的逆残差结构(the inverted residual with linear bottleneck)和SENet的基于squeeze and excitation结构的轻量级注意力模型。



**SE module 结构**：

（1）$F_{tr}$：$X$到$U$的卷积过程 ，但是通道之间的关系并没有发生变化；

（2）$F_{sq}$：将每个通道做了一个$squeeze$操作（主要是池化操作），将每个通道表示成了一个标量，得到per channel的描述；

（3）$F_{ex}$：将per channel标量进行“激活”，可以理解为算出了per channel的$W$（权值），实际上这一步就是全连接；

（4）最后将per channel的$W$（权重）乘回到原来的feature map上得到加权后的channel，将channel 做了恰当的融合；

SE-Module 可以用于网络的任意阶段，且**squeeze 操作保证了在网络的早期感受野就可以大到全图的范围。**

![SE module](/Users/momo/Documents/momo学习笔记/SE module.png)



**具体实践中的SE module有SE-inception Module and SE-ResNet Module:**

左图是SE-inception Module，第（2）步中的$squeeze$采用average pooling，得到$1*1*C$的向量；后面再接FC，但是为了减少参数，做了降维操作，增加了一个降维系数$r$，输出$1*1*\frac{C}{r}$；后接$ReLU$，再做一个升维操作，得到$1*1*C$，最终采用$Sigmoid$函数激活。激活之后，将每一个通道的权值向量$1*1*C$乘到相应的通道上，结构如下：(右图是resnet module，改造和inception分支很类似。)

![inception resnet SE module](/Users/momo/Documents/momo学习笔记/inception resnet SE module.png)





下图是MobileNetV3的结构：

![MobileNetV3](/Users/momo/Documents/momo学习笔记/MobileNetV3.png)

可以看出文章中提出的GhostNet模型结构用Ghost Bottlenecks代替了MobileNetV3中的Bottlenecks，剩下的模型中不同层的输入输出规模与MobileNetV3的基本一样。





**5.ShuffleNet**

**V1**

1.Channel Shuffle for Group Convolutions

（1）Xception和ResNeXt都没有考虑$1*1$（pointwise）的卷积的计算量，比如在ResNeXt中，组卷积只在$3*3$的层中使用，导致pointwise卷积占了Madds的93.4%；直观的解决方法就是在$1*1$的层中也使用组卷积；

（2）Channel Shuffle：对于前一组层的channel，先将所有的channel分为几个组，再将每个组中的通道划分为几个子组，然后用不同的子组来填充下一层中的每个组。这可以通过channel shuffle操作高效地实现（如下图）

![channel shuffle](/Users/momo/Documents/momo学习笔记/channel shuffle.png)



2.ShuffleNet Unit

![ShuffleNet Unit](/Users/momo/Documents/momo学习笔记/ShuffleNet Unit.png)

a)是MobileNetV2的bottleneck；b)是stride为1的ShuffleNet Unit，在$1*1$的pointwise卷积过程采用组卷积，以及Channel Shuffle，与shortcut的结果逐体素相加；c)是stride为2的ShuffleNet Unit，中间的DWConv步长为2， shortcut是一个步长为2的均值池化，两个结果进行一个channel concat。



3.Network Architecture

![ShuffleNet](/Users/momo/Documents/momo学习笔记/ShuffleNet.png)



**V2**

1.Practical Guidelines for Efficient Network Design

（1）Equal channel width minimizes memory access cost (MAC).卷积层的输入输出通道数相等时最节约存储空间；

（2）Excessive group convolution increases MAC.组卷积可以减少FLOPs，但是过度的组卷积增大存储空间；

（3）Network fragmentation reduces degree of parallelism.网络碎片化降低了并行度，应该减少分支；

（4）Element-wise operations are non-negligible.元素级别操作是不可忽略的，比如ReLU逐像素激活和depthwise convolutions。



2.ShuffleNet V2: an Efficient Architecture

ShuffleNetV2 Unit:

![ShuffleNetV2 Unit](/Users/momo/Documents/momo学习笔记/ShuffleNetV2 Unit.png)

（1）对于结构(c)：

1）在每个Unit开始，引入Channel Split操作，将输入的feature channels分为两个分支，分别有$c-c'$和$c$个channels；

2）其中一个分支直接恒等映射到下一层（G3），另一个分支经过三个卷积层，但是通道数不变（G1），三个卷积层中的$1*1$卷积不再像ShuffleNetV1中使用组卷积（G2），因为在channel split阶段已经分组了；

3）两个分支再经过concat融合，使得通道数和最开始的输入通道数一致（G1），最后再经过channel shuffle，加强两个分支的通道交互。

（2）对于结构(d)（经过unit(c)的channel shuffle之后，接上unit(d)），和ShuffleNetV1一样，提出一个下采样的unit。

在结构(c)(d)中，都没有ShuffleNetV1中的两分支add的操作，直接将两分支的结果进行concat，因为根据（G4），逐像素操作很费时间。

根据结构(c)(d)，构成ShuffleNetV2：（和ShuffleNetV1基本一致）

![ShuffleNetV2](/Users/momo/Documents/momo学习笔记/ShuffleNetV2.png)



**6.MnasNet(Mobile neural architecture search Net)**

1.Problem Formulation

（1）common method，$ACC(m)$代表模型准确率，$LAT(m)$代表模型的latency，$T$是目标latency限制，则优化问题简化如下：
$$
maximize_mACC(m)\\subject\,\,\,to\,\,\,LAT(m)\leq T
$$
（2）这种方法只是在优化单一指标，得不到多重帕累托最优解，帕累托最优是指模型有最高的准确率的同时，也有最低的latency。为了获得帕累托最优，改变优化问题如下：
$$
maximize_m ACC(m)\times[\frac{LAT(m)}{T}]^w\\w=\alpha\,\,if\,\,LAT(m)\leq T,\beta\,\,otherwise
$$
$\alpha，\beta$这样选择，在不同的acc-latency对应的模型之下，应该有相同的reward，如一个模型的准确率为$a$，latency为$l$，另一个模型的latency为$2l$，但是准确率提高了$5%$，此时应该有：$a(1+5\%)\cdot(\frac{2l}{T})^\beta=a\cdot(\frac{l}{T})$，求解得$\beta=-0.07$，再取$\alpha=\beta=-0.07$，后续模型都是采用这个参数，除非特殊说明。



2.Mobile Neural Architecture Search（移动端神经网络框架搜索）

2.1.Factorized Hierarchical Search Space（分解层次搜索空间）

以前的大多数方法只搜索几个复杂的单元，然后重复地堆叠相同的单元。与之前的方法不同，我们引入了一种新的分解层次搜索空间，它将CNN模型分解为独特的块，然后分别搜索每个块的操作数和连接数，从而允许在不同块中使用不同的层结构，如下图：

![MNasNet search](/Users/momo/Documents/momo学习笔记/MNasNet search.png)

将整个网络分为若干个block，每个block包含多个Layer。在搜索时，不同的block之间的结构是不同的，从相应的子空间中候选网络结构，选定之后，每个block里面的Layer是相同重复的。相比于之前单一搜索结构然后重复堆砌的方法，这种分层搜索的方法大大增加了层次多样性。

2.2.搜索算法：强化学习方法来搜索

具体的网络结构（MnasNet-A1）：

![MnasNet-A1](/Users/momo/Documents/momo学习笔记/MnasNet-A1.png)







**7.EfficientNet**

1.Compound Model Scaling（复合模型扩展）

1.1.模型数学表达

（1）实际中，卷积网络一般分为好几个阶段，但是每个阶段都是相同的结构；如ResNet有5个阶段，每个阶段都是一样的卷积操作，除了第一层有个下采样；

所以将一个卷积神经网络表示如下：
$$
N=\triangle_{i=1,...,s}F_i^{L_i}(X_{<H_i,W_i,C_i>})
$$
（2）和一般的卷积网络设计聚焦在找到最好的网络结构$F_i$不一样，model scaling要找到$L_i,C_i,H_i,W_i$之间合适的比例关系，而不改变baseline网络中的$F_i$结构；

（3）限定$F_i,L_i,H_i,W_i,C_i$不变，调整$w,d,r$系数（分别是通道数$C_i$，每阶段的卷积层数$L_i$以及空间尺寸$H_i,W_i$的比例系数）的值，使得模型获得最高的精度；

1.2.调整单一的参数

（1）增大网络的深度比例参数$d$可以获得更复杂的特征，但同时也会出现梯度消失的情形，尽管ResNet或者DenceNet中有skip connection的结构，但是ResNet-1000和ResNet-101的表现差不多；

（2）增大网络的宽度比例参数$w$可以使得网络更容易训练，获得更好的细粒度模式（fine-grained pattern），但是极度宽但是层数少的网络得不到高层次的表示特征，并且表现随着$w$的增大容易饱和；

（3）基础的分辨率是$224*224$，在一定程度上，提高分辨率系数$r$会提高表现，非常高的分辨率时增长幅度消失；

**总结：在一定范围内调整任何一个维度的参数都会适当提高模型表现，但是对于更大的模型，表现趋于饱和。**

1.3.复合调整参数

（1）增大图像分辨率时，相应的要增大模型深度（获得更好的表示特征）和模型宽度（获得更好的细粒度模式）；

**总结：调整模型参数$w,d,r$，要注意相互平衡。**



2.EfficientNet Architecture

（1）受到MnasNet的启发，开发了基线网络——通过利用多目标神经结构搜索优化准确性和FLOPs；

（2）网络的结构如下：（主要卷积模块为MBConv，和MnasNet一样）

![EifficientNet](/Users/momo/Documents/momo学习笔记/EifficientNet.png)

以此网络结构为baseline网络，应用上述的复合调整参数的方法来获得EfficientNet-B系列网络，步骤为：

1）先固定$\phi=1$，在小范围内搜索$\alpha,\beta,\gamma$的值使得满足FLOPs以及模型大小约束的条件下，模型准确率最大的最优解，最终得到$\alpha=1.2,\beta=1.1,\gamma=1.15$；

2）然后固定$\alpha,\beta,\gamma$，再选择$\phi$的值，获得EfficientNet-B0-B7的网络。



**8.模型结构对比总结**

**1.ResNet和MobileNetV1是最基本的两个trick，一个提出残差结构，引入shortcut；另一个将传统的卷积分离。后续的MobileNetV2,V3和GhostNet都采用了残差结构和卷积分离；**

**2.MobileNetV2在MobileNetV1的block之前添加了一层pointwise convolution来增加通道数，且添加了shortcut，和ResNet相比形状相反，相当于Inverted residuals；**

**3.MobileNetV3是综合了以下三种模型的思想：MobileNetV1的深度可分离卷积（depthwise separable convolutions）、MobileNetV2的具有线性瓶颈的逆残差结构(the inverted residual with linear bottleneck)和SENet的基于squeeze and excitation结构的轻量级注意力模型；**

**4.GhostNet与MobileNetV2网络的bottleneck非常类似，都采用了shortcut结构和深度分离卷积的结构，且整体上都呈现出逆残差的结构（即在bottleneck中，都是先扩大通道数，再减少通道数）。不同的是，MobileNetV2的bottleneck中，基本单元是pointwise和depthwise卷积，而GhostNet的bottleneck中的基本单元是ghost module结构，ghost module中包含了pointwise和depthwise卷积，并且有的bottleneck中还包含SE module；**

**5.GhostNet在MobileNetV3的基础上，将Bottlenecks替换成Ghost Bottlenecks，Ghost Bottlenecks本身也是一个残差结构，其中的ghost module也是传统卷积的另一种分解和加速；**

**6.ShuffleNetV1中提出的两个ShuffleNet Unit和GhostNet中的两个Ghost Bottleneck有点相似，stride分别是1和2，在stride为2的bottleneck中，shortcut都采用了下采样；ShuffleNetV2中的channel split，和ghost module中的固有channel很类似，都是把输入的channel的一部分channel直接恒等映射到下一层，另一部分再经过特定设计的卷积层作用；**

**7.对于所有的轻量级网络，MobileNet系列、ShuffleNet系列以及GhostNet都属于手动设计网络结构的小网络，MnasNet和EfficientNet属于神经网络框架自搜索的小网络，网络结构是在一个特定搜索空间中运用强化学习搜索算法搜索得到的结构。**



##### 二、比较模型的大小、parameters多少，以及FLOPs大小

**1.GhostNet的评测**

（1）计算模型大小和参数量

假设模型中有1M的parameters，每个参数以float32的形式存储，需要1M\*32个比特（bit），每8个构成一个字节（byte），所以实际文件大小是1M\*32/8=4M。在计算一个模型中的parameter的量时，直接模型文件实际大小除以4即得到parameters大小。parameters的大小也可以通过下面的工具来计算。

（2）计算FLOPs（同时也计算了parameters）

(参考https://github.com/nmhkahn/torchsummaryX)

计算代码：param_flops.py，计算结果截图如下：

![params and flops](/Users/momo/Documents/momo学习笔记/params and flops.png)

（3）测试GhostNet在ImageNet测试集上的分类精度

```shell
python test_time.py --evaluate
```

测试结果为：

![times_GhostNet](/Users/momo/Documents/momo学习笔记/times_GhostNet.png)

（4）pytorch模型GhostNet在5000张图像上的测试，代码test_time_pt.py

测试GhostNet的时间为：36.18s，单张图像的latency时间为7.23ms/--。



**2.MobileNet系列网络的评测**

该部分评测只测试了各个网络模型的单张图像的latency，其余指标采用官方数据。tensorflow官方发布的MobileNet系列网络运行时间的计算：

参考链接（https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb）

1.tensorflow模型MobileNet系列在5000张图像上的测试，代码test_time_tf2.py

分别测试MobileNetV1，MobileNetV2，MobileNetV3的FP32模型和INT8模型，结果分别为：time cost: 43.20 s，time cost: 50.119 s，57.36s/56.75s，单张图像的latency分别为8.64ms/--，10.02ms/--，11.47ms/11.35ms。



**3.ShuffleNet系列网络的评测**

测试代码eval_pytorch_model_image.py，测试ShuffleNetV1,V2两个网络模型的运行时间，结果分别为：92.76s和94.24s。



**4.MnasNet系列网络的评测**

参考链接：https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_example.ipynb，可以评测时间，具体见代码test_time_mnasnet.py。



**5.EfficientNet系列网络的评测**

先尝试了链接（https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite） 中的代码测试eval_ckpt_main.py测试时间，发现运行结果里每个模型（EfficientNet-lite0-lite2）单张运行图像时间在4s左右，和之前测试的MobileNet系列，GhostNet以及官方发布的时间不在一个量级，这段代码不能用于时间测试，测不出真实的latency。

再参考链接（https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification）， 从源码来运行.tflite模型，还是出现很多问题。



再使用亚勇哥给的一个脚本tflite_predict_file.py来测.tflite模型，步骤如下：

（1）在本地安装工具：https://github.com/lutzroeder/netron ，用于可视化.tflite模型各个节点的名称；

（2）获悉节点名称之后更改代码tflite_predict_file.py：

1）46行需要根据模型更改输入节点的名称；

```python
self.assertEqual('input', input_details[0]['name']) #源代码
self.assertEqual('images', input_details[0]['name']) # 替换之后
```

2）48行需要根据模型更改输入尺寸；

```python
self.assertTrue(([1, 224, 224, 3] == input_details[0]['shape']).all()) # 源代码
self.assertTrue(([1, 224, 224, 3] == input_details[0]['shape']).all()) # lite0不需要修改
self.assertTrue(([1, 240, 240, 3] == input_details[0]['shape']).all()) # lite1修改为240
self.assertTrue(([1, 260, 260, 3] == input_details[0]['shape']).all()) # lite2修改为260
```

3）49行均值及方差，用于quantization；

```python
self.assertEqual((0.007843137718737125, 128), input_details[0]['quantization']) # 源代码
self.assertEqual((0.007843137718737125, 128), input_details[0]['quantization']) # lite0_uint8

```

4）56行需要根据模型更改输出节点的名称；

```python
self.assertEqual('MobilenetV2/Predictions/Softmax', output_details[0]['name']) # 源代码
self.assertEqual('Softmax', output_details[0]['name']) # lite0_uint8
```

5）129行输出是int类型的0-255，需要除以255，更改为浮点型0-1之间的概率。注意83-89行以及126-132行



更改之后的脚本为test_tflite.py，该脚本在有的tensorflow版本中会报错，再次改写为test2_tflite.py，成功测试完EfficientNet系列模型（实际上该脚本不仅仅可以测试EfficientNet系列的网络，可以测试任意的tflite模型）。



移动端网络评测总结表格见（https://docs.google.com/spreadsheets/d/1Vqz9puFFImWB5TQHMKqnIkbSRmEXSMTpOgAnpvPgWSY/edit#gid=0）

**注：关于latency的测试，MobileNet,MnasNet,EfficientNet网络的latency都是单张图像在CPU上的时间，ShuffleNet,GhostNet网络的latency都是单张图像在GPU上的时间。**

