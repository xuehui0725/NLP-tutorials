---
title: NLP模型-05-transformer
date: 2020-05-02
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---

<!-- more -->


### 什么是transformer
一句话解释：`transformer`是带有`self-attention`的`seq2seq`，输出能同时计算，可以代替RNN结构（必须按输入的顺序计算），因此transformer对于sequence to sequence的应用场景更为高效。

transformer是首个完全抛弃RNN的recurrence，CNN的convolution，仅用attention来做特征抽取的模型。

**Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。**

<!-- more -->

### transformer结构解读
transformer的模型结构图如下：

<img src="https://i.loli.net/2020/07/12/SdtenNQgPkjm6iJ.jpg" width="40%" alt="transformer计算流程1" />

上图中，左半部分是Encoder，右半部分是Decoder。

1. Encoder
- `input`会通过`input embedding layer`变成一个`vector`
- 这个`vector`会加上`positional encoding`
- `positional encoding`接下来会进入灰色的block（灰色block会重复N次）
- 其中，灰色block的第一层是`Muti-head Attention`，sequence通过此层会得到另一个sequence
- 接下来通过`Add & Norm`，作用是将`Muti-Head Attention`的`output`和`Muti-head Attention`的`input`加起来再做layer normalization（和batch normalization不同）
- 接下来通过一个`Feed Forward layer`，会将sequence的每一个vector进行处理
- 然后再接上另一个`Add & Norm`

2. Decoder
- `outputs`会通过`output embedding layer`变成一个`vector`
- 这个`vector`会加上`positional encoding`
- `positional encoding`接下来会进入灰色的block（灰色block会重复N次）
- 其中，灰色block的第一层是`Masked Muti-head Attention`（masked：attend on the generated sequence），sequence通过此层会得到另一个sequence
- 接下来通过`Add & Norm`
- 再进入`Muti-Head Attention layer`接到之前`Encoder`部分的输出
- 接下来通过一个`Add & Norm`、`Feed Forward layer`、`Add & Norm`
- 再经过`Linear`、`Softmax`得到注重的输出。

**计算的具体流程如下图：**

第1步：根据Wq、Wk、Wv，计算每个vector对应的q、k、v，再根据右上角的公式计算αi

【PS】q、k、v分别代表Query向量（要去查询的，与每个词之间的关系），Key向量（等着被查的词）和Value向量（实际的特征信息），它们是通过3个不同的权值矩阵由嵌入向量X乘以三个不同的权值矩阵Wq、Wk、Wv得到

<img src="https://i.loli.net/2020/07/12/2N6PSW1F5ORb4zD.png" width="50%" alt="transformer计算流程1" />

第2步：对每个αi计算softmax值

<img src="https://i.loli.net/2020/07/12/LvVNyuGleXH84ba.png" width="50%" alt="transformer计算流程2" />

第3步：使用右上角的公式，计算每个vector对应的输出b

<img src="https://i.loli.net/2020/07/12/NLS2CcavgdDGwUA.png" width="50%" alt="transformer计算流程3" />

【注意】以上从x1到xn的运算是可以同时进行的，通过attention值的大小，可以与其他的vector产生上下文关联。

**transformer计算的范围**

Transformer的可处理文本长度定为512，而不是更大的值，一个很重要的因素是，在计算attention的过程中，需要计算（Multi-head attention并不会减少计算量），这也是为什么Transformer处理长距离依赖问题并不太好的原因之一。

### 概念详解

**multi-headed机制**（通常都是多层，一层的不够的）

多组Q、K、V能得到多组表达（类似CNN中的filter），总综合多种特征提取方案，得到综合的特征表达
- 通过不同的head得到多个特征表达（一般是8个）
- 将所有特征拼接在一起
- 可以通过再一层全连接来降维（乘上全连接矩阵）

<img src="https://i.loli.net/2020/07/13/FnO2htRbZYKJcux.png" width="90%" alt="multi-headed机制" />

**位置信息如何表达？**

加入了位置信息的编码，每个词的位置发生变化，那就不同了。位置信息编码的方法非常多，常见的是加上一个周期信号。

**什么是layer normalization？**

<img src="https://i.loli.net/2020/07/13/Wyf6jEcHhu5I9dQ.png" width="90%" alt="layer norm" />

LN 是在每一个样本上计算均值和方差，而不是BN那种在批方向计算均值和方差。

Add & Norm做了**残差连接**（多条路的选择给模型自己选择）

**decoder的特点**

- attention计算方式不同（不只有self-attention，还多了encoder-decoder attention）
- 加入了mask机制（decoder的输出只能一个个输出，需要依赖前面的vector，因此需要使用mask机制，就是不能预知未来输出的东西）
- 连softmax层，使用cross-entropy计算loss即可

### transformer为何计算速度快
我们可以将以上的计算过程使用一连串的矩阵乘法解决，而矩阵乘法可以轻易地使用GPU来加速。如下图所示：

**1. 每个vector的q、k、v的计算都可以使用矩阵相乘Wq*a、Wk*a、Wv*a加以计算：**

<img src="https://i.loli.net/2020/07/12/cV5HT1NZDBjloUf.png" width="50%" alt="transformer矩阵计算1" />

**2. α的计算也可以通过矩阵相乘表示不，如下图所示：**

（图一）

<img src="https://i.loli.net/2020/07/12/HkTUR4qDAmXW3C8.png" width="50%" alt="transformer矩阵计算2" />

（图二）

<img src="https://i.loli.net/2020/07/12/KWYHyGko3eEFTmB.png" width="50%" alt="transformer矩阵计算3" />

（图三）

<img src="https://i.loli.net/2020/07/12/ryFezKHq2cv5TJl.png" width="50%" alt="transformer矩阵计算4" />

**3. b的计算也可以通过矩阵相乘来解决：**

<img src="https://i.loli.net/2020/07/12/oM3ZXHFuBzD67Wn.png" width="50%" alt="transformer矩阵计算5" />

一下是综合矩阵运算的图解：

<img src="https://i.loli.net/2020/07/12/UzZWQCG64Et9y5P.png" width="50%" alt="transformer矩阵计算6" />

具体解析详见[NLP-06](https://iloveyou11.github.io/2020/04/05/NLP-06/)中的self-attention内容。

**attention整体计算流程：**
1. 每个词的Q会跟每个K计算得分（重要性越大，得分越高）
2. softmax后会得到整个加权结果（相当于归一化）
3. 此时每个词看的不只是他前面的序列，而是整个输入序列
4. 同一时间能计算出所有词的表达结果（利用矩阵乘法）

### 模型改进

原始版的Transformer虽然并不成熟，层数固定不够灵活、算力需求过大导致的不适合处理超长序列等缺陷限制了其实际应用前景。但是其优秀的特征抽取能力吸引了很多学者的关注。很多人提出了不同的变种Transformer来改进或者规避它的缺陷。其中，Universal Transformer、Transformer-XL、Reformer就是典型的代表。 

#### Universal Transformer

在Transformer中，输入经过Attention后，会进入全连接层进行运算，而Universal Transformer模型则会进入一个共享权重的transition function继续循环计算。这里Transition function可以和之前一样是全连接层，也可以是其他函数层。

为了控制循环的次数，模型引入了Adaptive Computation Time（ACT）机制。ACT可以调整计算步数，加入ACT机制的Universal transformer被称为Adaptive universal transformer。引入ACT机制后，模型对于文本中更重要的token会进行更多次数的循环，而对于相对不重要的单词会减少计算资源的投入。

<img src="https://i.loli.net/2020/07/22/JXgR19pBNy7xior.png" alt="Universal Transformer" width="80%" />

#### Transformer-XL

Transformer通常会将本文分割成长度小于等于d（默认是512）的segment，每个segment之间互相独立，互不干涉。距离超过512的token之间的依赖关系就完全无法建模抽取。同时，这还会带来一个`context fragmentation`的问题，因为segment的划分并不是根据语义边界，而是根据长度进行划分的，可能会将一个完整的句子拆成两个。

Transformer-XL提出了`segment-level Recurrence`来解决这个问题。在对当前segment进行处理的时候，缓存并利用上一个segment中所有layer的隐向量，而且上一个segment的所有隐向量只参与前向计算，不再进行反向传播。

Transformer-XL在没有大幅度提高算力需求的情况下，一定程度上解决了长距离依赖问题。

#### Reformer

针对transformer处理文本长度过短（512）的问题，提出了两个机制分别解决这两个问题，它们是`locality-sensitve hashing(LSH) attention`和`Reversible transformer`。

首先用LSH来对每个segment进行分桶，将相似的部分放在同一个桶里面。然后我们将每一个桶并行化计算其中两两之间的点乘。还考虑到有一定的概率相似的向量会被分到不同的桶里，因此采用了多轮hashing来降低这个概率。Reformer在减少了attention计算量的情况下，还减少了模型的内存占用，为未来大型预训练模型的落地奠定了基础。

---

推荐阅读：[【NLP】Transformer](http://mantchs.com/2019/09/26/NLP/Transformer/)