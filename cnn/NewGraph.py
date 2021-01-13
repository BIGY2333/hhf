import numpy as np
import tensorflow as tf
import copy
import math
from tensorflow.python.framework import graph_util
from tensorflow.contrib.layers import xavier_initializer
import math
import os
import LoadData
import sys
# Batch_size = 128
QP = 35
Iteration = 3001
alpha = 0.2
beta = 0.3
gamma = 0.5
Zreo_Flag = 2
epsilon = 0.001
decay = 0.99
REGULARIZATION_RATE = 0.01   #正则化项系数
LEARNING_RATE_DECAY = 0.99   #学习率的衰减率
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率
def Weight_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.03)
    # 使用了tf.truncted_normal产生随机变量来进行初始化：
    return tf.Variable(initial)

def Bias_Variable(shape):
    # 使用tf.constant常量函数来进行初始化：
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积操作
def Filter(input, w, sde_w, sde_h):
    return tf.nn.conv2d(input, w, [1, sde_w, sde_h, 1], padding="SAME")
    # input:a tensorflow ,shape:[Batch,Height,Width,Channel]
    # w: filter ,shape:[filter_height,filter_width,Channels,filter_numbers]
    # stride: vector ,shape:4*1 represent stride in every dimension
    # padding: "SAME" or "VALID" SAME:cover the edge of picture，use 0 for padding
    # output: a tensorflow ,shape:[Batch,Height,Width,Channels


def Pool_Layer(input, k_hei, k_wid, str_hei, str_wid):
    return tf.nn.max_pool(input, ksize=[1, k_hei, k_wid, 1], \
                          strides=[1, str_hei, str_wid, 1], padding="VALID")
    # input:a tensorflow ,feature map,shape:[Batch,Height,Width,Channel]
    # ksize: maxpool size shape,normal [1,Height,Width,1]
    # strides : vector ,shape:4*1 represent stride in every dimension
    # padding: "SAME" or "VALID" SAME:cover the edge of picture
    # output: a tensorflow ,shape:[Batch,Height,Width,Channels]


def Cal_Mulp(input, w, b):
    return tf.matmul(input, w) + b


def SPP(input,size,out_size):
    a1 = Pool_Layer(input,size,size,size,size)
    a2 = Pool_Layer(input,size*2,size*2,size*2,size*2)
    a3 = Pool_Layer(input,size*4,size*4,size*4,size*4)
    a1 = tf.reshape(a1,[-1,16*out_size])
    a2 = tf.reshape(a2,[-1,4*out_size])
    a3 = tf.reshape(a3,[-1,1*out_size])
    Vec1 = tf.concat([a1, a2], 1)
    Vec2 = tf.concat([Vec1, a3], 1)
    return  Vec2

def add_layer(inputs, in_szie, out_size, activate=None):
    weights = tf.Variable(tf.truncated_normal([in_szie, out_size], stddev=0.01))
    biases = tf.Variable(tf.truncated_normal([1, out_size], stddev=0.01))
    result = tf.matmul(inputs, weights) + biases
    if activate is None:
        return result
    else:
        return activate(result)
# 用于后续训练

def batch_norm_training(Conv1_1,mean,variance,shift,scale):

    batch_mean, batch_variance = tf.nn.moments(Conv1_1, [0, 1, 2], name=None, keep_dims=False)
    train_mean_1 = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
    train_variance_1 = tf.assign(variance, variance * decay + batch_variance * (1 - decay))

    with tf.control_dependencies([train_mean_1, train_variance_1]):
        return mean,variance,tf.nn.batch_normalization(Conv1_1, batch_mean, batch_variance, shift, scale, epsilon)

def batch_norm_inference(Conv1_1,mean,variance,shift,scale):

    return mean,variance,tf.nn.batch_normalization(Conv1_1, mean, variance, shift, scale, epsilon)


def Build_Network():
    # is_64 = tf.placeholder(tf.int16,name='m_64')
    # is_32 = tf.placeholder(tf.int16,name='m_32')
    ln = tf.placeholder(tf.float32,name='ln')
    Pool_para = 2
    # def m64():
    #     return tf.placeholder(tf.float32, (None, 64, 64, 1), name='X')
    # def m32():
    #     return tf.placeholder(tf.float32, (None, 32, 32, 1), name='X')
    # def m16():
    #     return tf.placeholder(tf.float32, (None, 16, 16, 1), name='X')
    # def judge():
    #     return tf.cond(is_32 > 0,lambda: m32(),lambda: m16())
    # def judge1(Pool2):
    #     return tf.cond(is_32 > 0, lambda: SPP(Pool2, 4, 32), lambda: SPP(Pool2, 2, 32))
    # def judge2(Pool4):
    #     return tf.cond(is_32 > 0, lambda: SPP(Pool4, 2, 32), lambda: SPP(Pool4, 1, 32))
    # X = tf.placeholder(tf.float32, (None, None, None, 1))
    # tf.image.ResizeMethod.NEAREST_NEIGHBOR这里的方法用的是最近邻差值
    # 这里是实用双线性差值
    # 使用tf.image系的函数要慎重，一定要check数据类型，check函数处理后是否在0-255的范围，尤其是resize相关。
    # X = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
    #                                 tf.image.ResizeMethod.BILINEAR+)
    X = tf.placeholder(tf.float32, (None, 32, 32, 1), name='X')
    y = tf.placeholder(tf.float32, [None, 1],name='y')
    is_training = tf.placeholder(tf.bool, name='is_train')
    kp = tf.placeholder(tf.float32, name='kp')
    #global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Conv1") as scope:
        kernel1_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.03), name="k1")

        b1_1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b1")
        Conv1_1 = tf.nn.bias_add(Filter(X, kernel1_1, 1, 1), b1_1, name='op_add1')
        bn1 = tf.layers.batch_normalization(Conv1_1, training=is_training)
        Output_Conv1_1 = tf.nn.relu(bn1)
        Pool1 = tf.nn.max_pool(Output_Conv1_1, ksize=[1, 2, 2, 1], \
                          strides=[1, 1, 1, 1], padding="SAME")
        kernel1_2 = tf.Variable(tf.truncated_normal([3, 3, 32,32 ], stddev=0.03), name="k2")

        # with tf.control_dependencies([assign_from_placeholder]):
        #     x_assign = x.assign(1)

        b1_2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b2")
        Output_Conv1_2 = tf.nn.bias_add(Filter(Pool1, kernel1_2, 1, 1), b1_2, name='op_add2')
        bn2 = tf.layers.batch_normalization(Output_Conv1_2, training=is_training)
        Output_Conv1_3 = tf.nn.relu(bn2)

        Pool2 = Pool_Layer(Output_Conv1_3, Pool_para, Pool_para, Pool_para, Pool_para)

    # l1 = tf.cond(is_64 > 0, lambda: SPP(Pool2, 8, 32), lambda: judge1(Pool2))

    with tf.name_scope("Conv2") as scope:
        kernel2_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.03), name="k1")
        # with tf.control_dependencies([assign_from_placeholder]):
        #     x_assign = x.assign(1)
        b2_1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b1")
        Conv2_1 = tf.nn.bias_add(Filter(Pool2, kernel2_1, 1, 1), b2_1, name='op_add1')
        bn3 = tf.layers.batch_normalization(Conv2_1, training=is_training)
        Output_Conv2_1 = tf.nn.relu(bn3)
        Pool3 = tf.nn.max_pool(Output_Conv2_1, ksize=[1, 2, 2, 1], \
                          strides=[1, 1, 1, 1], padding="SAME")
        kernel2_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.03), name="k2")
        b2_2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b2")
        Output_Conv2_2 = tf.nn.bias_add(Filter(Pool3, kernel2_2, 1, 1), b2_2, name='op_add2')
        bn4 = tf.layers.batch_normalization(Output_Conv2_2, training=is_training)
        Output_Conv2_3 = tf.nn.relu(bn4)
        Pool4 = Pool_Layer(Output_Conv2_3, Pool_para, Pool_para, Pool_para, Pool_para)

    # l2 = tf.cond(is_64 > 0, lambda: SPP(Pool4, 4,32), lambda: judge2(Pool4))
    # Vec1 = tf.concat([l1, l2], 1)
    l1 = tf.reshape(Pool4, [-1, 8*8*32])
    #weights1 = tf.Variable(tf.truncated_normal([1344, 128], stddev=0.01), name="w1")
    weights1=tf.get_variable('w1',[2048, 128], tf.float32, xavier_initializer())
    # weight_loss1 = tf.multiply(tf.nn.l2_loss(weights1), 0.001, name='weight_loss1')
    # tf.add_to_collection('losses', weight_loss1)
    b3 = tf.Variable(tf.truncated_normal([1, 128], stddev=0.01), name="b3")
    result1 = tf.matmul(l1, weights1) + b3
    Layer_1_pre = tf.nn.relu(result1)

    Layer_1_pre = tf.nn.dropout(Layer_1_pre, kp)

    #weights2 = tf.Variable(tf.truncated_normal([128, 8], stddev=0.01), name="w2")
    weights2 = tf.get_variable('w2',[128, 16], tf.float32, xavier_initializer())
    b4 = tf.Variable(tf.truncated_normal([1, 16], stddev=0.01), name="b4")
    result2 = tf.matmul(Layer_1_pre, weights2) + b4
    Layer_2_pre = tf.nn.relu(result2)
    Layer_2_pre = tf.nn.dropout(Layer_2_pre, kp)

    #weights3 = tf.Variable(tf.truncated_normal([8, 1], stddev=0.01), name="w3")
    weights3 = tf.get_variable( 'w3',[16, 1], tf.float32, xavier_initializer())
    b5 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.01), name="b5")
    result4 = tf.matmul(Layer_2_pre, weights3) + b5
    Pre_Label = tf.nn.sigmoid(result4)
    # tf.add_to_collection("pre64", Pre_64)
    Pre = tf.round(Pre_Label, name="output")
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # LOSS = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Pre, labels=y))
    LOSS = -tf.reduce_mean(y * tf.log(tf.clip_by_value\
                                                    (Pre_Label, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value \
                                                                                                    (1 - Pre_Label, 1e-10,
                                                                                                    1.0)))
    correct_Cu64_Num = tf.equal(Pre, y)
    # 把bool矩阵转换为float类型，并求平均即得到了准确率
    acc = tf.reduce_mean(tf.cast(correct_Cu64_Num, dtype=tf.float32),name='acc')
    #tf.add_to_collection('losses', LOSS)
   # LOSS_REGU = tf.add_n(tf.get_collection('losses'), name='total_loss')
   #  learning_rate = tf.train.exponential_decay(
   #      0.2,
   #      global_step=500*64,
   #      decay_rate=0.99,
   #      decay_steps=200,
   #      staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        Train = tf.train.AdamOptimizer(learning_rate=ln, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(LOSS)
    # Train = tf.group(Train_op, variables_averages_op)
    # Train = tf.train.AdamOptimizer(0.2).minimize(LOSS_REGU)

    return dict(
        inputs_Mat=X,
        ys_label=y,
        is_training=is_training,
        kp=kp,
        ln=ln,
        Train=Train,
        LOSS=LOSS,
        acc=acc,
        Pre=Pre,
        Pre_Label=Pre_Label,
    )