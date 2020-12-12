核心内容：文本摘要、命名实体识别、关系抽取、实体消歧、实体统一、指代消解、句法分析、CKY算法

本系列内容：

<img width="60%" src="https://i.loli.net/2020/03/24/ctd87JmD9RPTUq4.jpg" alt="系列4知识点" />

#### 文本摘要
1. 抽取式文本摘要
从文档中抽取其中一句话或者几句话构成摘要。
2. 生成式文本摘要
因为生成式文本摘要是一个端到端的过程，这种技术方案，近似于翻译任务和对话任务，从而可以吸收、借鉴翻译任务和对话任务的成过经验。
主要方法有：
- 早期的LSTM
- 早期的seq2seq
- seq2seq+attention模型
- self-attention和transform
- 预训练+微调，如Bert与xlnet

#### 命名实体识别（NER）
识别实体名称，如产品名、任命、公司名、组织名、机构名、学名等等

NER方法：
- 利用规则（比如正则匹配)
- 投票模型
- 利用分类模型（如时序模型：逻辑回归、SVM……时序模型：HMM、CRF、LSTM-CRF……）

#### 关系抽取
方法有：
- 基于规则
- 监督学习
- bootstrap（原始）
- bootstrap（snowball）
- distant supervision
- 无监督学习

##### bootstrap算法
bootstrap算法：“生成规则-生成tuple-生成规则-生成tuple”……迭代

<img width="60%" src="https://i.loli.net/2020/03/24/2vM8jmxgluOXybi.jpg" alt="bootstrap" />

bootstrap算法缺点：error accumulation（准确率不断下降）

<img width="40%" src="https://i.loli.net/2020/03/24/Xb15FAvrT64o9sK.jpg" alt="bootstrap缺点" />

##### snowball算法
在1）生成规则 2）生成tuple的基础上加了两步：3）评估规则准确率，过滤 4）评估tuple准确率，过滤

bootstrap是精准匹配，snowball是近似匹配

<img width="80%" src="https://i.loli.net/2020/03/24/TbWa6Gi3oCdM2wF.jpg" alt="snowball1" />

<img width="80%" src="https://i.loli.net/2020/03/24/ug7yz5xT3fLhEJs.jpg" alt="snowball2" />

<img width="80%" src="https://i.loli.net/2020/03/24/KVJ9qkF8pocTI3t.jpg" alt="snowball3" />

#### 实体消歧
同一单词或词语，在不同的上下文中，可能有不同的含义。

#### 实体统一
如何判断两个对象属于同一个实体？
方法：
1. 基于规则
2. 监督学习
3. 基于图的实体统一

#### 指代消解
他她它代表谁？代表哪个实体？还没有被解决的核心问题

#### 句法分析
##### CKY算法
CKY算法是一种使用动态规划对`上下文无关文法（CFG）`进行`语法分析（parsing）`的算法。

<img width="50%" src="https://i.loli.net/2020/03/24/Xty6sNZ3CnIRfjh.jpg" alt="CKY" />