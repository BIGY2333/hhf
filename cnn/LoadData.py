import numpy as np
import tensorflow as tf
import copy
import math
from tensorflow.python.framework import graph_util
import math
import os
import sys



#
# def Read_Train_Data():
#     # 文件序列
#     # 以下为读取文件阶码流程
#     filenames = [("Data/" + YUV + "/" + FILE_TRAIN + "%d.tfrecords" % i) for i in range(1, 28)]
#     filename_queue = tf.train.string_input_producer(filenames)
#     # Tensorflow需要队列流输入数据
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # tf.parse_single_example为解析器
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            "Mat": tf.FixedLenFeature([], tf.string),
#                                            "label_64": tf.FixedLenFeature([], tf.int64),
#                                            "label_32": tf.FixedLenFeature([], tf.string),
#                                            "label_16": tf.FixedLenFeature([], tf.string),
#                                        })
#
#     Mat_64 = tf.decode_raw(features["Mat"], tf.float32)
#     Mat_64 = tf.reshape(Mat_64, [64, 64, 1])
#     Label_32 = tf.decode_raw(features["label_32"], tf.float32)
#     Label_32 = tf.reshape(Label_32, [4])
#
#     Label_16 = tf.decode_raw(features["label_16"], tf.float32)
#     Label_16 = tf.reshape(Label_16, [16])
#
#     Label_64 = tf.cast(features["label_64"], tf.float32)
#     # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
#     #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
#     # tf.train.shuffle_batch（）这个函数对读取到的数据进行了batch处理，这样更有利于后续的训练。
#     Mat_batch, Label64_batch, Label32_batch, Label16_Batch = tf.train.shuffle_batch \
#         ([Mat_64, Label_64, Label_32, Label_16], batch_size=Batch_size, capacity=300, min_after_dequeue=5,
#          num_threads=1)
#     print("\n--------- finish reading train data-----------")
#     return Mat_batch, Label64_batch, Label32_batch, Label16_Batch
#
#
# def Read_Cross_Data():
#     # 文件序列
#     filenames = [("Data/" + YUV + "/" + FILE_TRAIN + "%d.tfrecords" % j) for j in range(1, 28)]
#     filename_queue = tf.train.string_input_producer(filenames)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            "Mat": tf.FixedLenFeature([], tf.string),
#                                            "label_64": tf.FixedLenFeature([], tf.int64),
#                                            "label_32": tf.FixedLenFeature([], tf.string),
#                                            "label_16": tf.FixedLenFeature([], tf.string),
#                                        })
#     Mat_64 = tf.decode_raw(features["Mat"], tf.float32)
#     Mat_64 = tf.reshape(Mat_64, [64, 64, 1])
#     Label_32 = tf.decode_raw(features["label_32"], tf.float32)
#     Label_32 = tf.reshape(Label_32, [4])
#
#     Label_16 = tf.decode_raw(features["label_16"], tf.float32)
#     Label_16 = tf.reshape(Label_16, [16])
#     Label_64 = tf.cast(features["label_64"], tf.float32)
#     # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
#     #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
#     Mat_batch, Label64_batch, Label32_batch, Label16_Batch = tf.train.shuffle_batch \
#         ([Mat_64, Label_64, Label_32, Label_16], batch_size=Batch_size, capacity=300, min_after_dequeue=5,
#          num_threads=1)
#     print("\n--------- finish reading cross data-----------\n")
#     return Mat_batch, Label64_batch, Label32_batch, Label16_Batch
#
#
# def Read_Test_Data():
#     # 文件序列
#     filenames = [("Data/" + YUV + "/" + FILE_CROSS + "%d.tfrecords" % j) for j in range(10, 20)]
#     filename_queue = tf.train.string_input_producer(filenames)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            "Mat": tf.FixedLenFeature([], tf.string),
#                                            "label_64": tf.FixedLenFeature([], tf.int64),
#                                            "label_32": tf.FixedLenFeature([], tf.string),
#                                            "label_16": tf.FixedLenFeature([], tf.string),
#                                        })
#     Mat_64 = tf.decode_raw(features["Mat"], tf.float32)
#     Mat_64 = tf.reshape(Mat_64, [64, 64, 1])
#     Label_32 = tf.decode_raw(features["label_32"], tf.float32)
#     Label_32 = tf.reshape(Label_32, [4])
#     Label_16 = tf.decode_raw(features["label_16"], tf.float32)
#     Label_16 = tf.reshape(Label_16, [16])
#     Label_64 = tf.cast(features["label_64"], tf.float32)
#     # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
#     #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
#     Mat_batch, Label64_batch, Label32_batch, Label16_Batch = tf.train.shuffle_batch \
#         ([Mat_64, Label_64, Label_32, Label_16], batch_size=300, capacity=300, min_after_dequeue=5, num_threads=1)
#     print("\n--------- finish reading test data ------------")
#     return Mat_batch, Label64_batch, Label32_batch, Label16_Batch
#
#     # 处理数据，每个像素点减去均值


def Normalize2(Mat, Width):
    for i in range(Width):
        for j in range(Width):
            Mat[i][j][0] = Mat[i][j][0] / 127.5 - 1
    return Mat


def Normalize(Mat):
    # Mat = Mat.reshape(4,1)
    # print(Mat)
    Mat_mean = np.mean(Mat)
    Mat_std = np.std(Mat)
    if Mat_std == 0:
        Mat = (Mat - Mat_mean)
        return Mat
    Mat = (Mat - Mat_mean) / Mat_std
    Nan_loc = np.isnan(Mat)
    Mat[Nan_loc] = 0
    # print("Mat mean is ",Mat_mean)
    # print("Mat std is ",Mat_std)
    return Mat


def Normalize1(Mat, Width):
    abe = Width * Width
    mean = 0
    sum = 0
    for i in range(Width):
        for j in range(Width):
            mean += Mat[i][j][0]
    mean = mean / (abe)
    for i in range(Width):
        for j in range(Width):
            sum += (Mat[i][j][0] - mean) ** 2
    std = sum / abe
    std = std ** 0.5
    if std == 0:
        for i in range(Width):
            for j in range(Width):
                Mat[i][j][0] = Mat[i][j][0] - mean
        return Mat
    for i in range(Width):
        for j in range(Width):
            Mat[i][j][0] = (Mat[i][j][0] - mean) / std
    return Mat
    # print("Mat mean is ",Mat_mean)
    # print("Mat std is ",Mat_std)


def Normalize_min(Mat, width):
    min1 = 1000
    max1 = -1000
    for i in range(width):
        for j in range(width):
            if min1 > Mat[i][j][0]:
                min1 = Mat[i][j][0]
            if max1 < Mat[i][j][0]:
                max1 = Mat[i][j][0]
    if min1 == max1:
        for i in range(width):
            for j in range(width):
                Mat[i][j][0] = 1
    else:
        for i in range(width):
            for j in range(width):
                Mat[i][j][0] = (Mat[i][j][0] - min1) / (max1 - min1)
    return Mat


def Normalize_Mat(Mat, s):
    for index, value in enumerate(Mat):
        Mat[index] = Normalize(value)
    return Mat

# def Normalize_Mat32(Mat):
#     for index,value in enumerate(Mat):
#         Mat[index][0:32,0:32] = Normalize(Mat[index][0:32,0:32],32)
#         Mat[index][0:32,32:64] = Normalize(Mat[index][0:32,32:64],32)
#         Mat[index][32:64,0:32] = Normalize(Mat[index][32:64,0:32],32)
#         Mat[index][32:64,32:64] = Normalize(Mat[index][32:64,32:64],32)
#     return Mat
#
# def Normalize_Mat16(Mat):
#     for index,value in enumerate(Mat):
#         Mat[index][0:16,0:16] = Normalize(Mat[index][0:16,0:16],16)
#         Mat[index][0:16,16:32] = Normalize(Mat[index][0:16,16:32],16)
#         Mat[index][0:16,32:48] = Normalize(Mat[index][0:16,32:48],16)
#         Mat[index][0:16,48:64] = Normalize(Mat[index][0:16,48:64],16)
#
#         Mat[index][16:32,0:16] = Normalize(Mat[index][16:32,0:16],16)
#         Mat[index][16:32,16:32] = Normalize(Mat[index][16:32,16:32],16)
#         Mat[index][16:32,32:48] = Normalize(Mat[index][16:32,32:48],16)
#         Mat[index][16:32,48:64] = Normalize(Mat[index][16:32,48:64],16)
#
#         Mat[index][32:48,0:16] = Normalize(Mat[index][32:48,0:16],16)
#         Mat[index][32:48,16:32] = Normalize(Mat[index][32:48,16:32],16)
#         Mat[index][32:48,32:48] = Normalize(Mat[index][32:48,32:48],16)
#         Mat[index][32:48,48:64] = Normalize(Mat[index][32:48,48:64],16)
#
#         Mat[index][48:64,0:16] = Normalize(Mat[index][48:64,0:16],16)
#         Mat[index][48:64,16:32] = Normalize(Mat[index][48:64,16:32],16)
#         Mat[index][48:64,32:48] = Normalize(Mat[index][48:64,32:48],16)
#         Mat[index][48:64,48:64] = Normalize(Mat[index][48:64,48:64],16)
#
#     return Mat