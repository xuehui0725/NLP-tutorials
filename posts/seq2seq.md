---
title: NLP模型-10-seq2seq
date: 2020-06-05
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


### seq2seq初识
当输入和输出都是不定长序列时，则可以使用`编码器-解码器`或`seq2seq模型`解决。

`seq2seq`属于`encoder-decoder`结构的一种，常见是`encoder-decoder`结构，基本思想就是利用两个RNN，一个RNN作为encoder，另一个RNN作为decoder，两个RNN网络是共同训练的。encoder负责将输入序列压缩成指定长度的向量，这个向量就可以看成是这个序列的语义，这个过程称为编码，获取语义向量最简单的方式就是直接将最后一个输入的隐状态作为语义向量C。也可以对最后一个隐含状态做一个变换得到语义向量，还可以将输入序列的所有隐含状态做一个变换得到语义变量。

而decoder则负责根据语义向量生成指定的序列，这个过程也称为解码，如下图，最简单的方式是将encoder得到的语义变量作为初始状态输入到decoder的RNN中，得到输出序列。可以看到上一时刻的输出会作为当前时刻的输入，而且其中语义向量C只作为初始状态参与运算，后面的运算都与语义向量C无关。

seq2seq的应用场景如下：
1. 机器翻译（当前最为著名的Google翻译，就是完全基于Seq2Seq+Attention机制开发出来的）。
2. 聊天机器人（小爱，微软小冰等也使用了Seq2Seq的技术（不是全部））。
3. 文本摘要自动生成（今日头条等使用了该技术）。
4. 图片描述自动生成。
5. 机器写诗歌、代码补全、生成 commit message、故事风格改写等。

#### 编码器
编码器的作⽤是把⼀个不定⻓的输⼊序列变换成⼀个定⻓的背景变量`c`，并在该背景变量中编码输⼊序列信息，常用的编码器是RNN网络。

编码器可以使用单向的RNN，也可以使用双向的RNN，单向RNN的每个时间步的隐藏状态只取决于该时间步及之前的输⼊⼦序列，双向RNN的每个时间步的隐藏状态同时取决于该时间步之前和之后的⼦序列。

#### 解码器
将上⼀时间步的输出`Yt-1`以及背景`x`作为输⼊，将上一步的隐藏状态`St-1`转化为当前时间步的隐藏状态`St`。基于`St-1`、`St`、`c`，使⽤⾃定义的输出层和softmax运算来计算当前时间步输出`y`的概率分布。

### seq2seq与VAE/AE的区别
#### seq2seq
主流的Seq2Seq都是基于Encoder-Decoder来实现的，Encoder-Decoder也不只是能用于Seq2Seq场景。s2s是具体的模型，encoder-decoder是模型设计“范式”。

#### AE
AE是AutoEncoder（自编码器）的简称，自动编码器是一种数据的压缩算法，`自编码器（Autoencoder）是神经网络的一种，经过训练后能尝试将输入复制到输出（先将输入转化为隐层，再将隐层转化为输出）。`

自编码内部有一个隐藏层 h，可以产生编码（code）表示输入。该网络可以看作由两部分组成：一个由函数 h = f(x) 表示的编码器和一个生成重构的解码器 r = g(h)。如果一个自编码器只是简单地学会将处处设置为 g(f(x)) = x，那么这个自编码器就没什么特别的用处。

#### VAE
VAE是变分自编码器，相比于自编码器，VAE更倾向于数据生成。只要训练好了decoder，我们就可以从标准正态分布生成数据作为解码器的输入，来生成类似但不同于训练数据的新样本，作用类似GAN。

`VAE相当于针对隐层h加了noise，variance应用了noise的大小。`

### 实战演练
通过sin与con进行叠加变形生成无规律的模拟曲线，使用Seq2Seq模式对其进行学习，拟合特征，从而达到可以预测下一时刻数据的效果。

定义两个曲线sin和con，通过随机值将其变形偏移，将两个曲线叠加：
```python

import random
import math
        
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

def do_generate_x_y(isTrain, batch_size, seqlen):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sin_data = amp_rand * np.sin(np.linspace(
            seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2)  )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig_data = amp_rand * np.cos(np.linspace(
            seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2)) + sin_data

        batch_x.append(np.array([ sig_data[:seqlen] ]).T)
        batch_y.append(np.array([ sig_data[seqlen:] ]).T)

    # shape: (batch_size, seq_length, output_dim)
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y

#生成15个连续序列，将con和sin随机偏移变化后的值叠加起来
def generate_data(isTrain, batch_size):
    seq_length =15
    if isTrain :
        return do_generate_x_y(isTrain, batch_size, seq_length)
    else:
        return do_generate_x_y(isTrain, batch_size, seq_length*2)
    
        
sample_now, sample_f = generate_data(isTrain=True, batch_size=3)
print("training examples : ")
print(sample_now.shape)
print("(seq_length, batch_size, output_dim)")


seq_length = sample_now.shape[0]
batch_size = 10

output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12  
layers_num = 2

# Optmizer:
learning_rate =0.04
nb_iters = 100

lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
        
        
tf.reset_default_graph()

encoder_input = []
expected_output = []
decode_input =[]
for i in range(seq_length):
    encoder_input.append( tf.placeholder(tf.float32, shape=( None, input_dim)) )
    expected_output.append( tf.placeholder(tf.float32, shape=( None, output_dim)) )
    decode_input.append( tf.placeholder(tf.float32, shape=( None, input_dim)) )

    
tcells = []
for i in range(layers_num):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))
Mcell = tf.contrib.rnn.MultiRNNCell(tcells)

dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_input,decode_input,Mcell)

reshaped_outputs = []
for ii in dec_outputs :
    reshaped_outputs.append( tf.contrib.layers.fully_connected(ii,output_dim,activation_fn=None))


# L2 loss
output_loss = 0
for _y, _Y in zip(reshaped_outputs, expected_output):
    output_loss += tf.reduce_mean( tf.pow(_y - _Y, 2) )
   
# generalization capacity)
reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ("fully_connected" in tf_var.name ):
        #print(tf_var.name)
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

loss = output_loss + lambda_l2_reg * reg_loss
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)   

sess = tf.InteractiveSession()
        
def train_batch(batch_size):

    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})

    c =np.concatenate(( [np.zeros_like(Y[0])],Y[:-1]),axis = 0)

    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    c =np.concatenate(( [np.zeros_like(Y[0])],Y[:-1]),axis = 0)#来预测最后一个序列
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})    
    output_lossv,reg_lossv,loss_t = sess.run([output_loss,reg_loss,loss], feed_dict)
    print("-----------------")    
    print(output_lossv,reg_lossv)
    return loss_t


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t % 50 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,nb_iters, train_loss, test_loss))
print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))        
        
        
        
# Plot loss over time:
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0, len(test_losses))) /
    float(len(test_losses) - 1) * (len(train_losses) - 1),
    np.log(test_losses),label="Test loss")
    
plt.plot(np.log(train_losses),label="Train loss")
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()        
        
        
# Test
nb_predictions = 5
print("visualize {} predictions data:".format(nb_predictions))

preout =[]
X, Y = generate_data(isTrain=False, batch_size=nb_predictions)
print(np.shape(X),np.shape(Y))
for tt in  range(seq_length):
    feed_dict = {encoder_input[t]: X[t+tt] for t in range(seq_length)}
    feed_dict.update({expected_output[t]: Y[t+tt] for t in range(len(expected_output))})
    c =np.concatenate(( [np.zeros_like(Y[0])],Y[tt:seq_length+tt-1]),axis = 0)  #从前15个的最后一个开始预测  

    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
    preout.append(outputs[-1])

print(np.shape(preout))#将每个未知预测值收集起来准备显示出来。
preout =np.reshape(preout,[seq_length,nb_predictions,output_dim])

for j in range(nb_predictions):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = X[:, j, k]
        expected = Y[seq_length-1:, j, k]#对应预测值的打印

        pred = preout[:, j, k]

        label1 = "past" if k == 0 else "_nolegend_"
        label2 = "future" if k == 0 else "_nolegend_"
        label3 = "Pred" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected) + len(past)),
                 expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)),
                 pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions vs. future")
    plt.show()
```