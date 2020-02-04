import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from genplate import GenPlate, gen_sample, chars
from utils.ops import *

def create_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_tfrecord(filename, x_name='image', y_name='label', x_shape=[72, 272, 3], y_shape=[7], batch_size=64,
                  shuffle_data=False, num_threads=1):
    """
    读取TFRecord文件
    :param filename:
    :param x_name: 给定训练用x的名称
    :param y_name: 给定训练用y的名称
    :param x_shape: x的格式
    :param y_shape: y的格式
    :param batch_size: 批大小
    :param shuffle_data: 是否混淆数据，如果为True，那么进行shuffle操作
    :param num_threads: 线程数目
    :return:
    """
    # 获取队列
    filename_queue = tf.train.string_input_producer([filename])
    # 构建数据读取器
    reader = tf.TFRecordReader()
    # 读取队列中的数据
    _, serialized_example = reader.read(filename_queue)

    # 处理样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            x_name: tf.FixedLenFeature([], tf.string),
            y_name: tf.FixedLenFeature([], tf.string)
        }
    )

    # 读取特征
    image = tf.decode_raw(features[x_name], tf.float32)
    label = tf.decode_raw(features[y_name], tf.int32)

    # 格式重定
    image = tf.reshape(image, x_shape)
    label = tf.reshape(label, y_shape)

    # 转换为批次的Tensor对象
    capacity = batch_size * 6 + 10
    if shuffle_data:
        image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                              num_threads=num_threads, min_after_dequeue=int(capacity / 2))
    else:
        image, label = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity, num_threads=num_threads)

    return image, label


def model_mobile_v2(inputs, num_classes, bn_train= True):


    exp = 3

    with tf.variable_scope('Mobile_v2', initializer=tf.truncated_normal_initializer(stddev=0.1)):
        net = conv2d_block(inputs, 32, 3, 1, bn_train, name='conv1_1')
        # [N，72, 272, 32]
        """
        input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=False
        """

        net = res_block(net, 1, 16, 1, bn_train, 'res2_1')
        # [N, 72, 272, 16]

        net = res_block(net, exp, 24, 2, bn_train, 'res3_1' )
        net = res_block(net, exp, 24, 2, bn_train, 'res3_2')
        # [N, 36, 136, 24]

        net = res_block(net, exp, 32, 2, bn_train, 'res4_1')
        net = res_block(net, exp, 32, 2, bn_train, 'res4_2')
        net = res_block(net, exp, 32, 2, bn_train, 'res4_3')
        # [N, 18, 68, 32]

        net = res_block(net, exp, 64, 1, bn_train, 'res5_1')
        net = res_block(net, exp, 64, 1, bn_train, 'res5_2')
        net = res_block(net, exp, 64, 1, bn_train, 'res5_3')
        net = res_block(net, exp, 64, 1, bn_train, 'res5_4')
        # [N, 18, 68, 64]

        net = res_block(net, exp, 96, 1, bn_train, 'res6_1')
        net = res_block(net, exp, 96, 1, bn_train, 'res6_2')
        net = res_block(net, exp, 96, 1, bn_train, 'res6_3')
        # [N, 18, 68, 96]

        net = res_block(net, exp, 160, 2, bn_train, 'res7_1')
        net = res_block(net, exp, 160, 2, bn_train, 'res7_2')
        net = res_block(net, exp, 160, 2, bn_train, 'res7_3')
        # [N, 9, 34, 160]

        net = res_block(net, exp, 320, 1, bn_train, 'res8_1')
        # [N, 9, 34, 320]

        net = pwise_block(net, 1280, bn_train, 'conv9_1')
        # [N, 9, 34, 1280]
        net = global_avg(net)
        # [N, 1, 1, 1280]


        # 做7个模型
        with tf.variable_scope('fc1'):
            logit1 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc2'):
            logit2 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc3'):
            logit3 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc4'):
            logit4 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc5'):
            logit5 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc6'):
            logit6 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        with tf.variable_scope('fc7'):
            logit7 = flatten(conv_1x1(net, num_classes, 'logits', bias=True))

        return logit1, logit2, logit3, logit4, logit5, logit6, logit7



def losses(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):

    labels = tf.convert_to_tensor(labels, tf.int32)
    with tf.variable_scope('loss1'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit1, labels=labels[:, 0])
        loss1 = tf.reduce_mean(ce1, name='loss1')

    with tf.variable_scope('loss2'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=labels[:, 1])
        loss2 = tf.reduce_mean(ce1, name='loss2')

    with tf.variable_scope('loss3'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=labels[:, 2])
        loss3 = tf.reduce_mean(ce1, name='loss3')

    with tf.variable_scope('loss4'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit4, labels=labels[:, 3])
        loss4 = tf.reduce_mean(ce1, name='loss4')


    with tf.variable_scope('loss5'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit5, labels=labels[:, 4])
        loss5 = tf.reduce_mean(ce1, name='loss5')


    with tf.variable_scope('loss6'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit6, labels=labels[:, 5])
        loss6 = tf.reduce_mean(ce1, name='loss6')


    with tf.variable_scope('loss7'):
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit7, labels=labels[:, 6])
        loss7 = tf.reduce_mean(ce1, name='loss7')

    return loss1, loss2, loss3, loss4, loss5, loss6, loss7


def create_optimizer(loss1, loss2, loss3, loss4, loss5, loss6, loss7, lr=0.01):

    with tf.variable_scope('optimizer1'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt1 = optimizer.minimize(loss1)

    with tf.variable_scope('optimizer2'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt2 = optimizer.minimize(loss2)


    with tf.variable_scope('optimizer3'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt3 = optimizer.minimize(loss3)

    with tf.variable_scope('optimizer4'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt4 = optimizer.minimize(loss4)

    with tf.variable_scope('optimizer5'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt5 = optimizer.minimize(loss5)

    with tf.variable_scope('optimizer6'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt6 = optimizer.minimize(loss6)


    with tf.variable_scope('optimizer7'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_opt7 = optimizer.minimize(loss7)

    return train_opt1, train_opt2, train_opt3, train_opt4, train_opt5, train_opt6, train_opt7


def create_accuracy(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
    """
    计算准确率
    :param logit1:
    :param logit2:
    :param logit3:
    :param logit4:
    :param logit5:
    :param logit6:
    :param logit7:
    :param labels:
    :return:
    """
    logits_all = tf.concat([logit1, logit2, logit3, logit4, logit5, logit6, logit7],axis=0)

    # 标签转置
    labels = tf.convert_to_tensor(labels, tf.int32)

    labels_all = tf.reshape(tf.transpose(labels), [-1])

    # 计算准确率
    with tf.name_scope('accuracy'):
        correct_pred = tf.nn.in_top_k(logits_all, labels_all, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy

def train(train_tfrecord_name, img_h=72, img_w=272, img_channel=3, batch_size=4, max_epochs=3000,lr=0.001):

    # 定义持久化文件
    checkpoint_dir = './model/checkpoint'
    create_dir_path(checkpoint_dir)

     # 读取tfrecord的文件
    image, label = read_tfrecord(train_tfrecord_name, batch_size=batch_size, shuffle_data=True)

    bn_training = tf.placeholder_with_default(True, shape=None, name='bn_training')

    logit1, logit2, logit3, logit4, logit5, logit6, logit7 = model_mobile_v2(image, num_classes=65, bn_train= bn_training)

     # 定义损失
    loss1, loss2, loss3, loss4, loss5, loss6, loss7 = losses(logit1, logit2, logit3, logit4, logit5, logit6, logit7, label)

     # 构建优化器
    train_opt1, train_opt2, train_opt3, train_opt4, train_opt5, train_opt6, train_opt7 = create_optimizer(loss1, loss2, loss3, loss4, loss5, loss6, loss7, lr)

     # 计算准确率
    accuracy = create_accuracy(logit1, logit2, logit3, logit4, logit5, logit6, logit7, label)

    # 构建持久化对象
    saver = tf.train.Saver(max_to_keep=1)
    checkpoint_dir = './models/mnist'
    create_dir_path(checkpoint_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


    #启动相关线程

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(1,max_epochs):
            _, _, _, _, _, _, _, loss1_, loss2_, loss3_, loss4_, loss5_, loss6_, loss7_, accuracy_=sess.run([train_opt1, train_opt2, train_opt3, train_opt4, train_opt5, train_opt6, train_opt7,loss1, loss2, loss3, loss4, loss5, loss6, loss7,accuracy])

            if step % 2 ==0:
                avg_loss = loss1_ + loss2_ + loss3_ + loss4_ + loss5_ + loss6_ + loss7_
                print("step:{} - Train_loss:{:.5f} - Train acc:{}".format(step, avg_loss, accuracy_))


    # coord.request_stop()
    # coord.join(threads)

if __name__ == '__main__':
    tfrecord_dir_path = 'D:/python_test/深度学习/20191215__AI20__CarCode/车牌识别生成的tf_record数据/tf_record'
    train_tfrecord_path = os.path.join(tfrecord_dir_path, "train_data.tfrecord")
    train(train_tfrecord_path)






