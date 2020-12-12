---
title: NLP模型-09-textRNN&textCNN
date: 2020-06-01
categories: NLP
author: yangpei
comments: true
cover_picture: /images/banner.jpg
---


### TextRNN
#### 模型结构
TextRNN是利用RNN网络来解决文本分类问题，当然，也可以采用RNN的变种LSTM、GRU来实现……

文本分类的任务有：
- 垃圾邮件分类
- 情感分析
- 新闻/报道主题分析
- 问答系统中的问句分类
- ……

1. 结构1
流程：embedding—>BiLSTM—>concat final output/average all output----->softmax layer

<img src="https://i.loli.net/2020/07/28/vwRmjkPyd9n2rCc.png" alt="TextRNN1" width="60%" />

取前向/反向LSTM在最后一步的隐藏状态进行拼接，经过softmax层进行多分类；或者是取前向/反向LSTM在每一步上的隐藏状态，对每一步的两个隐藏状态进行拼接，然后对所有步拼接后的隐藏状态取平均值，然后再经过softmax层进行多分类，

2. 结构2
流程：embedding–>BiLSTM---->(dropout)–>concat ouput—>UniLSTM—>(droput)–>softmax layer

<img src="https://i.loli.net/2020/07/28/kH46Et2SXnhgyKC.png" alt="TextRNN2" width="80%" />

与结构1不同，结构2是在双向LSTM的基础上又叠加了一个单向的LSTM，把双向LSTM在每一步上的两个隐藏状态进行拼接，再进行dropout操作，作为上层单向LSTM每一步长上的一个输入，最后取上层单向LSTM最后一步上的隐藏状态，再进行dropout操作，最后经过一个softmax层进行一个多分类。

[中文文本分类之TextRNN](https://www.cnblogs.com/Luv-GEM/p/10836454.html)

【总结】
textRNN的结构十分灵活，可以任意改变。如：
- 把LSTM改为GRU
- 把双向改为单向
- 添加dropout/BN或者堆叠多层

textRNN在文本分类任务上可以取得很好的效果。

#### 代码实战

##### 数据集下载
使用THUCNews的一个[子集](https://www.lanzous.com/i5t0lsd)进行训练与测试，本次训练使用了其中的10个分类，每个分类6500条数据。类别分别为——体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐。

**cnews_loader.py为数据的预处理文件**
```
read_file(): 读取文件数据;
build_vocab(): 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
read_vocab(): 读取上一步存储的词汇表，转换为{词：id}表示;
read_category(): 将分类目录固定，转换为{类别: id}表示;
to_words(): 将一条由id表示的数据重新转换为文字;
process_file(): 将数据集从文字转换为固定长度的id序列表示;
batch_iter(): 为神经网络的训练准备经过shuffle的批次的数据。
```

##### textRNN模型和可配置的参数
```python
from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from rnn_model import TRNNConfig, TextRNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
```
##### 开始训练
```python
type_ = 'train'

print('Configuring RNN model...')
config = TRNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
model = TextRNN(config)

if type_ == 'train':
    train()
else:
    test()
```
##### 模型测试
```python
test()
```

### TextCNN

#### 模型结构

可以将⽂本当作⼀维图像，从而可以⽤⼀维卷积神经⽹络来捕捉临近词之间的关联。将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。

TextCNN的详细流程：
- `Embedding`：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
- `Convolution`：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
- `MaxPolling`：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
- `FullConnection and Softmax`：最后接一层全连接的 softmax 层，输出每个类别的概率。

TextCNN的模型结构如下：

<img src="https://i.loli.net/2020/07/28/4zd5ngprFLqZoSO.png" alt="TextCNN" width="90%" />

#### 代码实战
##### 数据集下载
使用THUCNews的一个[子集](https://www.lanzous.com/i5t0lsd)进行训练与测试，本次训练使用了其中的10个分类，每个分类6500条数据。类别分别为——体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐。

**cnews_loader.py为数据的预处理文件**
```
read_file(): 读取文件数据;
build_vocab(): 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
read_vocab(): 读取上一步存储的词汇表，转换为{词：id}表示;
read_category(): 将分类目录固定，转换为{类别: id}表示;
to_words(): 将一条由id表示的数据重新转换为文字;
process_file(): 将数据集从文字转换为固定长度的id序列表示;
batch_iter(): 为神经网络的训练准备经过shuffle的批次的数据。
```

##### textCNN的模型和可配置的参数
```python
from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
```
##### 开始训练
```python
type_ = 'train'

print('Configuring CNN model...')
config = TCNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
model = TextCNN(config)

if type_ == 'train':
    train()
else:
    test()
```
##### 模型测试
```python
test()
```