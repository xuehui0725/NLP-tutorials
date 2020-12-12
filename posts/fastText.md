---
title: NLP模型-08-fastText
date: 2020-05-21
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


#### fastText初始
fastText是一个快速文本分类算法，FastText 算法能获得和深度模型相同的精度，但是计算时间却要远远小于深度学习模型。fastText 可以作为一个文本分类的 baseline 模型。fastText不需要训练好的词向量模型，它会自己训练词向量模型。fastText两个重要的优化：`Hierarchical Softmax`、`N-gram`。fastText模型架构和word2vec中的CBOW很相似， 不同之处是fastText预测标签而CBOW预测的是中间词，模型架构类似但是模型任务不同。

使用fastText进行文本分类时，会产生单词的embedding，当然，我们也可以用预训练好的embedding来训练fastText模型。

【核心思想】将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。

#### n-gram表示单词

fastText模型架构:其中x1,x2,…,xN−1,xN表示一个文本中的`n-gram`向量，每个特征是词向量的平均值。这和前文中提到的cbow相似，cbow用上下文去预测中心词，而此处用全部的n-gram去预测指定类别。如下图所示：

<img src="https://i.loli.net/2020/07/28/RLT4XoQv86d1Mif.png" alt="fastText" width="50%" />

采用n-gram表示单词有以下几点好处：
1. 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。
2. 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。

#### 层次softmax

**为什么要使用层次softmax模型？**
在标准的softmax中，计算一个类别的softmax概率时，我们需要对所有类别概率做归一化，在这类别很大情况下非常耗时，因此提出了分层softmax(Hierarchical Softmax),思想是根据类别的频率构造霍夫曼树来代替标准softmax，通过分层softmax可以将复杂度从N降低到logN，下图给出分层softmax示例：

```
这里补充一下什么是haffman树：
给定N个权值作为N个叶子结点，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。

haffman树的构造过程：
假设有n个权值，则构造出的哈夫曼树有n个叶子结点。 n个权值分别设为 w1、w2、…、wn，则哈夫曼树的构造规则为：
(1) 将w1、w2、…，wn看成是有n 棵树的森林(每棵树仅有一个结点)；
(2) 在森林中选出两个根结点的权值最小的树合并，作为一棵新树的左、右子树，且新树的根结点权值为其左、右子树根结点权值之和；
(3)从森林中删除选取的两棵树，并将新树加入森林；
(4)重复(2)、(3)步，直到森林中只剩一棵树为止，该树即为所求得的哈夫曼树。
```

<img src="https://i.loli.net/2020/07/28/HwRrnAq7vzYgX3a.png" alt="层次softmax" width="50%" />

#### 与CBOW的不同
1. CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；
2. CBOW的输入单词被one-hot编码过，fastText的输入特征是被embedding过；
3. CBOW的输出是目标词汇，fastText的输出是文档对应的类标。

#### fastText优势
1. 适合于大型数据和高效快速计算的场景
2. 支持多语言，如英语、德语、西班牙语、法语以及捷克语等多种语言，性能要比当下的word2vec工具好不少
3. 专注于文本分类任务，如文本倾向性分析或标签预测方面都会得了当下最好的表现