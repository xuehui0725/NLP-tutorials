---
title: NLP模型-11-attention
date: 2020-06-12
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


### 何为Attention

<img width="60%" src="https://i.loli.net/2020/03/31/NF5q9JroSgapxTt.png" alt="attention" />

`Attention`是一种用于提升基于RNN（LSTM或GRU）的`Encoder + Decoder`模型的效果的的机制。`Attention`目前非常流行，广泛应用于机器翻译、语音识别、图像标注等很多领域，之所以它这么受欢迎，是因为`Attention`给模型赋予了区分辨别的能力，例如，在机器翻译、语音识别应用中，为句子中的每个词赋予不同的权重，使神经网络模型的学习变得更加灵活。

`Attention` 机制的本质来自于人类视觉注意力机制。人们视觉在感知东西的时候一般不会是一个场景从到头看到尾每次全部都看，而往往是根据需求观察注意特定的一部分。而且当人们发现一个场景经常在某部分出现自己想观察的东西时，人们会进行学习在将来再出现类似场景时把注意力放到该部分上。

仍然以循环神经⽹络为例，注意⼒机制通过对编码器所有时间步的隐藏状态做加权平均来得到背景变量。解码器在每⼀时间步调整这些权重，即注意⼒权重，从而能够在不同时间步分别关注输⼊序列中的不同部分并编码进相应时间步的背景变量。

在计算 `Attention` 时主要分为三步：
1. 第一步是将 query 和每个 key 进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等
2. 第二步一般是使用一个 softmax 函数对这些权重进行归一化
3. 最后将权重和相应的键值 value 进行加权求和得到最后的 `Attention`

**Attention机制的基本思想是**，打破了传统编码器-解码器结构在编解码时都依赖于内部一个固定长度向量的限制。Attention机制的实现是通过保留LSTM编码器对输入序列的中间输出结果，然后训练一个模型来对这些输入进行选择性的学习并且在模型输出时将输出序列与之进行关联。换一个角度而言，输出序列中的每一项的生成概率取决于在输入序列中选择了哪些项。

### self-attention
#### 原理
`self attention`也经常被称为intra Attention（内部Attention），最近一年也获得了比较广泛的使用，比如Google最新的机器翻译模型内部大量采用了`self attention`模型。

`self-attention` 可以是一般 `Attention` 的一种特殊情况，在 `self-attention` 中，Q=K,V 每个序列中的单元和该序列中所有单元进行 `Attention` 计算。

Google 提出的多头 `Attention` 通过计算多次来捕获不同子空间上的相关信息。`self-attention` 的特点在于无视词之间的距离直接计算依赖关系，能够学习一个句子的内部结构，实现也较为简单并行可以并行计算。
从一些论文中看到，`self-attention` 可以当成一个层和 RNN，CNN，FNN 等配合使用，成功应用于其他 NLP 任务。

#### 计算流程
1. 首先，`self-attention`会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64。
<img width="60%" src="https://i.loli.net/2020/03/31/UEuWVo8qeidIX4h.jpg" alt="self attention1" />

2. 计算`self-attention`的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点成，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2。
<img width="60%" src="https://i.loli.net/2020/03/31/t1CbX8recRGfVpv.jpg" alt="self attention2" />

3. 接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大。
<img width="60%" src="https://i.loli.net/2020/03/31/Un4A5FkGJ6BDChs.jpg" alt="self attention3" />

4. 下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵 Q 与 K 相乘，乘以一个常数，做softmax操作，最后乘上 V 矩阵。
<img width="60%" src="https://i.loli.net/2020/03/31/xDhEYdRyCwU97NG.jpg" alt="self attention4" />

#### 优势
引入`self-attention`后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是`self-attention`在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，`self-attention`对于增加计算的并行性也有直接帮助作用。这是为何`self-attention`逐渐被广泛使用的主要原因。

含注意⼒机制的变换器的编码结构在后来的BERT预训练模型中得以应⽤并令后者⼤放异彩：微调后的模型在多达11项⾃然语⾔处理任务中取得了当时最先进的结果。除了⾃然语⾔处理领域，注意⼒机制还被⼴泛⽤于图像分类、⾃动图像描述、唇语解读以及语⾳识别。