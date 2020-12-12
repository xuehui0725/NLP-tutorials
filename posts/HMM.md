---
title: NLP模型-02-HMM
date: 2020-03-30
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


### 初识HMM

HMM是`Hidden Markov Model`的缩写，是一个时序类的模型，`x`是观测值，`z`是隐变量。`z`可以视为状态，每个状态`z`都可以产生一个观测值`x`。HMM是有方向的生成模型。HMM的基础结构如下：

<img src="https://i.loli.net/2020/07/23/DJw7436Us1KPmcY.png" alt="HMM结构" width="80%" />

HMM中三个不同的参数：`θ=(A,B,Π)`，其中：
1. `A`是`状态z到状态z的转移矩阵`，指从一个状态z转移到下一个状态z的概率。
2. `B`是`状态z到结果x的生成矩阵`，表示从一个状态z下有多大概率生成某观测值x。但是只有在离散型随机变量的情况下，才能写成是矩阵的形式，如果为连续性随机变量，则需要使用到GMM（高斯混合模型）。
3. `Π`是`状态z首次出现概率向量`，表示状态Z1到状态Zn出现在第一个位置的概率。

如下图所示，`A`、`B`均为矩阵形式，`Π`为向量形式。

<img src="https://i.loli.net/2020/07/23/MPQpLulqTESiIna.png" alt="HMM参数" width="80%" />

针对此模型，主要有以下两大任务：
1. 给定模型的参数`θ`，找出最合适的`Z`
2. 估计模型的参数`θ`

#### 求解最好的z
1. 方法1：列举出所有的可能序列，可能很容易去评估likelihood，直接计算`P(Z1)*P(Z2|Z1)*P(Z3|Z2)……P(Zn|Zn-1) * P(X1|Z1)*P(X2|Z2)*P(X3|Z3)……P(Xn|Zn)`即可。

其中，`P(Z1)*P(Z2|Z1)*P(Z3|Z2)……P(Zn|Zn-1)`表示`状态z`序列转移的总概率，`P(X1|Z1)*P(X2|Z2)*P(X3|Z3)……P(Xn|Zn)`表示从`状态z1`到`状态zn`生成各个`x`的总概率。最后，对比穷举的所有可能`z`序列的概率，选择出最大的即可，结果就是最合适的`z`。

<img src="https://i.loli.net/2020/07/23/SZCaVv1dRiUm94h.png" alt="穷举求解z" width="80%" />

2. 方法2：使用维特比算法（Viterbi）解决——核心是动态规划的思想

如何在无穷多的路径中去选择最优的路径？将大问题转化为小问题，使用动态规划解决——每一个状态z下的最佳路径，都是选取的上一个`z`的最佳路径加上新路径总和的最优值，即`DP(z)=max(DP(z-1)+P(z-1,z))`。如下图所示：

<img src="https://i.loli.net/2020/07/22/Km61k5VhQCiFrbG.png" alt="Viterbi算法求解z1" width="80%" />
<img src="https://i.loli.net/2020/07/23/u2F3g6MdRLOlBpW.png" alt="Viterbi算法求解z2" width="80%" />

#### 估计参数θ
1. **F/B算法**：给定任何一个时间k，能计算出Zk属于某一个具体状态的概率的多少

<img src="https://i.loli.net/2020/07/23/FSLubr2DBfO7MU6.png" alt="FB1" width="80%" />

<img src="https://i.loli.net/2020/07/23/IiFvN2HzCuWtLVd.png" alt="FB2" width="80%" />

<img src="https://i.loli.net/2020/07/23/uempVrbYcNCqPSj.png" alt="FB3" width="80%" />

2. **Forward算法**：给定1-k时刻所有x的值，得到隐变量Zk的值

<img src="https://i.loli.net/2020/07/23/OoeWsm4UqixM7QP.png" alt="forwrad" width="80%" />

3. **Backward算法**：给定Zk，求出下一时刻X(k+1)的值

<img src="https://i.loli.net/2020/07/23/hKXt9uCsGLMUH4v.png" alt="backward" width="80%" />

### 三大参数计算

分为以下两种情况：`complete case`和`incomplete case`，其中，`complete case`指x（观测值）和z（隐变量）均已知的情况，`incomplete case`指x（观测值）已知，但是z（隐变量）未知的情况，这时必须借助其他算法（如EM算法）来解决。

<img src="https://i.loli.net/2020/07/23/wxigFXIh7uojKAQ.png" alt="complete vs incomplete" width="80%" />

#### complete case

Z和X均已知

<img src="https://i.loli.net/2020/07/23/8WTsrduVnmjCowg.png" alt="complete case" width="80%" />

1. Π等于每个状态Z出现在第一个位置上的概率，直接通过图即可求解出来（因为Z已知），组成一个向量。
2. A矩阵是通过每个状态转化为另一个状态的概率计算出来的，矩阵的维度和状态个数相等。计算过程也是非常简单，直接计算有多少次从状态Zi转移到了状态Zj，再计算概率即可。
3. B矩阵是通过每个状态生成某个观测值的概率计算出来的。计算过程也是非常简单，直接计算有多少次从状态Zi生成了观测值Xi，再计算概率即可。

#### incomplete case

X已知，Z未知，需要使用到EM算法，均使用概率估计（不能确定每个观测值对应隐变量z出现的必然性，均使用概率值表示）

<img src="https://i.loli.net/2020/07/23/9DTrp3KQRHw2uBS.png" alt="incomplete case" width="80%" />

1. **估计Π**

解决的问题：计算每个状态z首次出现的概率

假定z共有3种状态1、2、3，则分别计算：
```
P(z1=1|x1)  P(z1=2|x1)  P(z1=3|x1)
P(z1=1|x2)  P(z1=2|x2)  P(z1=3|x2)
P(z1=1|x3)  P(z1=2|x3)  P(z1=3|x3)
```
再计算向量`T=[P(z1=1|x1)+P(z1=1|x2)+P(z1=1|x3),P(z1=2|x1)+P(z1=2|x2)+P(z1=2|x3),P(z1=3|x1)+P(z1=3|x2)+(z1=3|x3)]`，最后对T进行归一化得到向量`Π`。具体流程如下图所示：

<img src="https://i.loli.net/2020/07/23/ZfL86wXR73Y1vdU.png" alt="估计PI2" width="80%" />

2. **估计B**

解决的问题：计算每个状态z生成不同x的概率矩阵

假设z有3个状态1、2、3，x有3种可能观测值a、b、c，我们可以计算出1、2、3分别生成a、b、c的出现次数，构建3X3矩阵，最后进行行向量归一化操作，可以得到矩阵B

<img src="https://i.loli.net/2020/07/22/hbC1jyBJRNe6DGo.png" alt="估计B" width="80%" />

3. **估计A**

解决的问题：计算每个状态z之间的转移概率矩阵

如下图所示，可以计算得到每个隐变量中z的分布概率，因此我们可以得到z的不同变量之间转化的概率只和，最后进行归一化操作即可。

<img src="https://i.loli.net/2020/07/23/yugKLJhUoPcrOaS.png" alt="估计A3" width="80%" />