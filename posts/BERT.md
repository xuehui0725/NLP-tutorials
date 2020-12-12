---
title: NLP模型-06-Bert
date: 2020-05-08
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


Bert（Bidirectional Encoder Representations from Transformers）说白了就是`transformer`中的`encoder`部分，就是怎么将transformer模型中的特征学习出来，是Word2Vec的替代者。

1. 使用了Transformer作为算法的主要框架，能更彻底的捕捉语句中的双向关系；
2. 使用了Mask Language Model(MLM) 和 Next Sentence Prediction(NSP) 的多任务训练目标；
3. 使用更强大的机器训练更大规模的数据，使BERT的结果达到了全新的高度，并且Google开源了BERT模型，用户可以直接使用BERT作为Word2Vec的转换矩阵并高效的将其应用到自己的任务中。

**bert的缺点：**
1. 测试数据是没有mask token的，训练数据与测试数据不匹配
2. 缺乏生成能力
3. 针对每个mask预测时没有考虑相关性（互相独立）

#### 什么是预训练模型
假设我们有大量的维基百科数据，那么我们可以用这部分巨大的数据来训练一个泛化能力很强的模型，当我们需要在特定场景使用时，例如做文本相似度计算，那么，只需要简单的修改一些输出层，再用我们自己的数据进行一个增量训练，对权重进行一个轻微的调整。预训练的好处在于在特定场景使用时不需要用大量的语料来进行训练，节约时间效率高效，bert就是这样的一个泛化能力较强的预训练模型。

BERT预训练模型的输出结果，无非就是一个或多个向量。下游任务可以通过精调（改变预训练模型参数）或者特征抽取（不改变预训练模型参数，只是把预训练模型的输出作为特征输入到下游任务）两种方式进行使用。

#### 如何训练Bert
BERT是一个多任务模型，它的任务是由两个自监督任务组成，即MLM和NSP。

**任务1：Masked Language Model（MLM）**
在训练的时候随机从输入预料上mask掉一些单词（有15%的词汇被随机mask掉），然后通过的上下文预测该单词。交给模型自己去预测被mask的是什么，类似于完形填空。

**任务2：Next Sentence Prediction（NSP）**
预测两个句子是否应该连在一起，句子B是否是句子A的下文。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。

<img src="https://i.loli.net/2020/07/13/4HJVnyOFmdNA9Eh.png" width="80%" alt="bert-句子判断" />

#### 举例说明
例如阅读理解题，输入文章和问题看，输出答案的位置。

如何设计网络呢？需要分别计算答案的起始位置和终止位置，如下图所示：

<img src="https://i.loli.net/2020/07/13/9wmjBbWXN5pTseY.png" width="80%" alt="bert-阅读" />

#### Bert模型如何使用

打开github/bert，下载pre-trained models，在此基础上做fine-tuning操作。

建议练手github/bert上列举的项目。
1. download github/bert的代码，放入新建的项目文件夹（`git clone https://github.com/google-research/bert`）
2. 根据不同的任务，选择下载预训练模型，放入checkpoint文件夹
3. 下载数据集，如glue-data
4. 开始训练
5. 在output文件下面，可以找到对应模型参数文件，就是训练的结果

#### Bert模型的变种
##### XLNet
`先将句子打乱，再复原句子。`XLNet融合了GPT和Bert两个模型，等效为mask机制+序列预测的难度加强版。

1. GPT是自编码模型， 通过双向LSTM编码提取语义。
2. Bert是自回归模型，不分顺序，用attention加强提取语义的效果。
3. XLNet融合了Bert的结构和GPT的有向预测。

<img src="https://i.loli.net/2020/07/16/wJriGV7cQYLtfNq.png" alt="XLNet" width="80%" />

XLNet解决了Bert的以下缺点：

<img src="https://i.loli.net/2020/07/17/iWoAPZ1TOuDqhnk.png" alt="XLNet改进" width="80%" />

##### SpanBert

对应论文：《Improving Pre-training by Representing and Predicting Spans》
`预测一个范围内的所有词，而不是像Bert一样预测一个个的词。`

1. 提出了更好的 `Span Mask` 方案，SpanBERT 不再对随机的单个 token 添加掩膜，而是对随机对邻接分词添加掩膜；
2. 通过加入 Span Boundary Objective (SBO) 训练目标，通过使用分词边界的表示来预测被添加掩膜的分词的内容，不再依赖分词内单个 token 的表示，增强了 BERT 的性能，特别在一些与 Span 相关的任务，如抽取式问答；
3. 用实验获得了和 XLNet 类似的结果，发现不加入 Next Sentence Prediction (NSP) 任务，直接用连续一长句训练效果更好。

<img src="https://i.loli.net/2020/07/16/t84cMzRsrFAhoJT.png" alt="SpanBert" width="80%" />

##### T5
T5是text-to-text transfer transformer）的简称，其实T5简单的说就是将所有 NLP 任务都转化成 Text-to-Text （文本到文本）任务。

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
[github地址](https://github.com/google-research/text-to-text-transfer-transformer)
`所有的任务都转化为序列到序列的任务`

##### StructBert

[论文](https://arxiv.org/pdf/1908.04577.pdf)

`打断某段词语顺序，预测时按正常输出，还能预测两个句子的关系（前后、后前、无关系）`

StructBert在Bert原有的MaskLM的训练目标上，增加了两个基于语言结构的训练目标：词序(word-level ordering)和句序(sentence-level ordering)任务。

给定句子对(S1, S2)，判断S2是否是S1的下一个句子，或上一个句子，或毫无关联的句子（从NSP的0/1分类变成了三分类问题）。

<img src="https://i.loli.net/2020/07/16/Y3ILtDFR7MkczoK.png" alt="StructBert" width="80%" />

##### FastBert

在每层Transformer后都去预测样本标签，如果某样本预测结果的置信度很高，就不用继续计算了。论文把这个逻辑称为样本自适应机制（Sample-wise adaptive mechanism），就是自适应调整每个样本的计算量，容易的样本通过一两层就可以预测出来，较难的样本则需要走完全程。

以下是论文提出的FastBert模型：

<img src="https://i.loli.net/2020/07/16/mgYp7IFyKun9OoD.png" alt="FastBert" width="100%" />

具体流程：
1. 获取某一版的Pre-training Bert.
2. 最后一层加上任务所需的分类网络，对Bert进行fine-tune。
3. 开始蒸馏，固定Bert主干的参数。在每一层decoder后都加上分类网络，注意这些网络是不共享参数的，各分各的。利用最后一层输出的分类概率，作为中间每一层分类的目标分布，利用KL散度或JS散度做分布拟合。

##### VL-Bert

[论文](https://arxiv.org/abs/1908.08530)

VL-Bert是一种新型的通用视觉-语言预训练模型（Visual-Linguistic BERT，简称 VL-BERT），该模型采用简单而强大的 Transformer 模型作为主干网络，并将其输入扩展为同时包含视觉与语言输入的多模态形式，适用于绝大多数视觉-语言下游任务。

<img src="https://i.loli.net/2020/07/16/LDcEvnGglPxHiYO.png" alt="VL-Bert" width="80%" />

> VL-BERT 的主干网络使用 TransformerAttention 模块，并将视觉与语言嵌入特征作为输入，其中输入的每个元素是来自句子中的单词、或图像中的感兴趣区域（RoIs）。在模型训练的过程中，每个元素均可以根据其内容、位置、类别等信息自适应地聚合来自所有其他元素的信息。在堆叠多层 TransformerAttention 模块后，其特征表示即具有更为丰富的聚合与对齐视觉和语言线索的能力。

任务：
1. 屏蔽语言模型（Masked Language Modeling），即随机屏蔽掉语句中的一些词，并预测当前位置的词是什么；
2. 屏蔽 RoI 分类（MaskedRoIClassification），即随机屏蔽掉视觉输入中的一些 RoIs，并预测此空间位置对应 RoI 的所属类别；
3. 图像标题关联预测（Sentence-Image Relationship Prediction），即预测图像与标题是否属于同一对。

##### VideoBert

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766)

作者们借鉴了语言建模中十分成功的 BERT 模型，在它的基础上进行改进，从视频数据的向量量化和现有的语音识别输出结果上分别导出视觉 token 和语言学 token，然后在这些 token 的序列上学习双向联合分布。作者们在多项任务中测试了这个模型，包括动作分类和视频描述。

##### DynaBERT

论文：《DynaBERT: Dynamic BERT with Adaptive Width and Depth》

论文中作者提出了新的训练算法，同时对不同尺寸的子网络进行训练，通过该方法训练后可以在推理阶段直接对模型裁剪。依靠新的训练算法，本文在效果上超越了众多压缩模型，比如DistillBERT、TinyBERT以及LayerDrop后的模型。

论文对于BERT的压缩流程是这样的：
- 训练时，对宽度和深度进行裁剪，训练不同的子网络
- 推理时，根据速度需要直接裁剪，用裁剪后的子网络进行预测

#### 对于Bert的思考

> BERT适用场景
`第一，如果NLP任务偏向在语言本身中就包含答案，而不特别依赖文本外的其它特征，往往应用Bert能够极大提升应用效果。`典型的任务比如QA和阅读理解，正确答案更偏向对语言的理解程度，理解能力越强，解决得越好，不太依赖语言之外的一些判断因素，所以效果提升就特别明显。反过来说，对于某些任务，除了文本类特征外，其它特征也很关键，比如搜索的用户行为／链接分析／内容质量等也非常重要，所以Bert的优势可能就不太容易发挥出来。再比如，推荐系统也是类似的道理，Bert可能只能对于文本内容编码有帮助，其它的用户行为类特征，不太容易融入Bert中。
`第二，Bert特别适合解决句子或者段落的匹配类任务。`就是说，Bert特别适合用来解决判断句子关系类问题，这是相对单文本分类任务和序列标注等其它典型NLP任务来说的，很多实验结果表明了这一点。而其中的原因，我觉得很可能主要有两个，一个原因是：很可能是因为Bert在预训练阶段增加了Next Sentence Prediction任务，所以能够在预训练阶段学会一些句间关系的知识，而如果下游任务正好涉及到句间关系判断，就特别吻合Bert本身的长处，于是效果就特别明显。第二个可能的原因是：因为Self Attention机制自带句子A中单词和句子B中任意单词的Attention效果，而这种细粒度的匹配对于句子匹配类的任务尤其重要，所以Transformer的本质特性也决定了它特别适合解决这类任务。
`第三，Bert的适用场景，与NLP任务对深层语义特征的需求程度有关。`感觉越是需要深层语义特征的任务，越适合利用Bert来解决；而对有些NLP任务来说，浅层的特征即可解决问题，典型的浅层特征性任务比如分词，POS词性标注，NER，文本分类等任务，这种类型的任务，只需要较短的上下文，以及浅层的非语义的特征，貌似就可以较好地解决问题，所以Bert能够发挥作用的余地就不太大，有点杀鸡用牛刀，有力使不出来的感觉。
这很可能是因为Transformer层深比较深，所以可以逐层捕获不同层级不同深度的特征。于是，对于需要语义特征的问题和任务，Bert这种深度捕获各种特征的能力越容易发挥出来，而浅层的任务，比如分词／文本分类这种任务，也许传统方法就能解决得比较好，因为任务特性决定了，要解决好它，不太需要深层特征。
`第四，Bert比较适合解决输入长度不太长的NLP任务，而输入比较长的任务，典型的比如文档级别的任务，Bert解决起来可能就不太好。`主要原因在于：Transformer的self attention机制因为要对任意两个单词做attention计算，所以时间复杂度是n平方，n是输入的长度。如果输入长度比较长，Transformer的训练和推理速度掉得比较厉害，于是，这点约束了Bert的输入长度不能太长。所以对于输入长一些的文档级别的任务，Bert就不容易解决好。结论是：Bert更适合解决句子级别或者段落级别的NLP任务。          出自[《一文读懂BERT(原理篇)》](https://blog.csdn.net/jiaowoshouzi/java/article/details/89073944)

#### Bert的应用
涵盖了Question Answer(QA，问答系统)与阅读理解、搜索与信息检索（IR）、对话系统／聊天机器人（Dialog System or Chatbot）、文本摘要、数据增强、文本分类、序列标注……详见[《Bert时代的创新（应用篇）：Bert在NLP各领域的应用进展》](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410022&idx=1&sn=40fd5ec1b428073020af50bd6e991c32&scene=21#wechat_redirect)