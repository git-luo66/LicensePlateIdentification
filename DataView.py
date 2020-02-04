
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from genplate import GenPlate, gen_sample, chars

def make_example(image, label):
    """
    产生Example对象
    :param image:
    :param label:
    :return:
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))


def generate_TFRecord(filename, genplate, height=72, weight=272, num_plat=1000):
    """
    随机生成num_plat张车牌照并将数据输出形成TFRecord格式
    :param filename: TFRecord格式文件存储的路径
    :param genplate: 车牌照生成器
    :param height: 车牌照高度
    :param weight: 车牌照宽度
    :param num_plat: 需要生成车牌照的数量
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num_plat):
        num, img = gen_sample(genplate, weight, height)
        # TODO: 因为MxNet中的格式要求导致的问题，必须转换回[height, weight, channels]
        img = img.transpose(1, 2, 0)
        img = img.reshape(-1).astype(np.float32)
        num = np.array(num).reshape(-1).astype(np.int32)

        ex = make_example(img.tobytes(), num.tobytes())
        writer.write(ex.SerializeToString())
    writer.close()


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

img_h = 72
img_w = 272
channels = 3
num_label = 7
batch_size = 1

tfrecord_dir_path = 'D:/python_test/深度学习/20191215__AI20__CarCode/车牌识别生成的tf_record数据/tf_record'
train_tfrecord_path = os.path.join(tfrecord_dir_path, "train_data.tfrecord")

# 从磁盘中读取数据（构建了读取数据的操作符）
train_image, train_label = read_tfrecord(train_tfrecord_path,
                                         x_shape=[img_h, img_w, channels],
                                         y_shape=[num_label],
                                         batch_size=batch_size,
                                         shuffle_data=True)
# print(train_image, train_label)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 2. 获取一下数据看看
    # 2. 启动相关的线程
    coor = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coor)

    for _ in range(3):
        _image, _label = sess.run([train_image, train_label])
        print(_image.shape, _label.shape, type(_image), _image.max(), _image.min())
        # print(_label)

        imgs = 255 * np.squeeze(_image)
        print(imgs.shape, _label.shape, type(imgs), imgs.max(), imgs.min())

        imgs = np.uint8(imgs)
        print(imgs.shape, _label.shape, type(imgs), imgs.max(), imgs.min())
        print('*' * 29)
        print(imgs.shape)  # (1, 72, 272, 3)

        plt.imshow(imgs)
        plt.show()

        # cv2.imshow('img', imgs)
        # cv2.waitKey(50000)
        # cv2.destroyAllWindows()
        """
        (2, 72, 272, 3) (2, 7)
        [[20 53 31 60 32 59 36]
         [17 62 60 50 42 51 56]]
        """