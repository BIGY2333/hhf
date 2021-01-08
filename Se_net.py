import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import time
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 8 # how many split ?
blocks = 3 # res_block ! (split + transition)
depth = 64 # out channel

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

reduction_ratio = 4

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 100


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

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network


def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units=1, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Global_Average_Pooling(x):
    return tf.reduce_mean(x, [1, 2], keep_dims=True)

class SE_ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x


    def Build_SEnet(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        # x = self.residual_layer(x, out_dim=256, layer_num='3')

        x = Global_Average_Pooling(x)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x



def train(Batch_size,graph):
    filenames = ["./train_1.tfrecords"]
    # filenames_cross = ["Data/SPP测试数据/train_1_64_cross.tfrecords"]
    filename_queue = tf.train.string_input_producer(filenames)
    WIDTH = 32
    # Tensorflow需要队列流输入数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # tf.parse_single_example为解析器
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "Mat": tf.FixedLenFeature([], tf.string),
                                           "label": tf.FixedLenFeature([], tf.int64),
                                       })

    Mat_32 = tf.decode_raw(features["Mat"], tf.float64)
    Mat_32 = tf.reshape(Mat_32, [WIDTH, WIDTH, 1])

    Label_32 = tf.cast(features["label"], tf.float32)

    # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
    #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
    # tf.train.shuffle_batch（）这个函数对读取到的数据进行了batch处理，这样更有利于后续的训练。

    Mat_batch, Label32_batch = tf.train.shuffle_batch \
        ([Mat_32, Label_32], batch_size=Batch_size, capacity=3 *  Batch_size, min_after_dequeue=5,
         num_threads=32)
    print("\n--------- finish reading train data-----------")

    with tf.Session() as sess:
        # 开启协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        # 一定要开启队列，要不然是空的
        threads = tf.train.start_queue_runners(sess, coord)
        initglo = tf.global_variables_initializer()
        initloc = tf.local_variables_initializer()
        sess.run(initglo)
        sess.run(initloc)
        tf.local_variables_initializer().run()
        epoch = 1
        learning_rate = 0.01
        for j in range(epoch):
            if j % 3 == 0:
                learning_rate = learning_rate * 0.1
            print(learning_rate)
            acc = []
            ite = []
            ite.append(0)
            acc.append(0)
            a = 0
            for i in range(test_iteration):
                # 这句话的意思是从队列里拿出指定批次的数据
                is_training = True
                mat_32, label_32 = sess.run([Mat_batch, Label32_batch])
                mat_32 = mat_32.reshape(Batch_size, 32, 32, 1)
                # print("Mat_64 is ",Mat_64)
                # print("Label_64 is ",Label_64)
                mat_32 = mat_32.astype('float32')
                label_32 = label_32.astype("float32")
                label_32 = label_32.reshape(Batch_size, 1)

                for b in range(Batch_size):
                    mat_32[b] = Normalize(mat_32[b])


                LOSS_32, acc_rate32, Pre_lable_32, _ = sess.run(
                    [graph["LOSS"], graph["acc"], graph["Train"]],
                    feed_dict={
                        graph["inputs_Mat"]: mat_32,
                        graph["ys_label"]: label_32,
                        graph['is_training']: is_training,
                        graph['learning_rate']: learning_rate,
                    })
                if i % 100 == 0:
                    is_training = False
                    print("\n------------- Iteration %d --------------" % (i))
                    # print("LOSS64 is ", LOSS_64)
                    # print("LOSS32 is ", LOSS_32)
                    print("LOSS16 is ", LOSS_32)
                    # print("acc_rate64 is : ", acc_rate64)
                    # print("acc_rate32 is : ", acc_rate32)
                    print("acc_rate16 is : ", acc_rate32)

    # filename_queue_cross = tf.train.string_input_producer(filenames_cross)
    # reader_cross = tf.TFRecordReader()
    # _, serialized_example_cross = reader_cross.read(filename_queue_cross)
    # features_cross = tf.parse_single_example(serialized_example_cross,
    #                                          features={
    #                                              "Mat": tf.FixedLenFeature([], tf.string),
    #                                              "label": tf.FixedLenFeature([], tf.int64),
    #                                          })
    #
    # Mat_32_cross = tf.decode_raw(features_cross["Mat"], tf.float64)
    # Mat_32_cross = tf.reshape(Mat_32_cross, [WIDTH, WIDTH, 1])
    #
    # Label_32_cross = tf.cast(features_cross["label"], tf.float32)
    #
    # Mat_batch_cross, Label32_batch_cross = tf.train.shuffle_batch \
    #     ([Mat_32_cross, Label_32_cross], batch_size=Batch_size,
    #      capacity=Batch_size * 3, min_after_dequeue=5,
    #      num_threads=32)



def Build_Network():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    label = tf.placeholder(tf.float32, shape=[None, 1])

    training_flag = tf.placeholder(tf.bool)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = SE_ResNeXt(x, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return dict(
        inputs_Mat=x,
        # qp=qp,
        ys_label=label,
        is_training=training_flag,
        Train=train,
        LOSS=l2_loss,
        acc=accuracy,
        learning_rate=learning_rate,
        #Pre_Label=Pre_Label,
    )

if __name__ == "__main__":
    demoStart = time.time()
    # G0 = NewGraph.Build_Network_16()
    G1 = Build_Network()
    train(32,G1)
    demoEnd = time.time()
    print("Total time cost : ", demoStart - demoEnd)

#
# x = np.array([[[1.,2.,3.],[4.,5.,6.]],[[1.,2.,3.],[4.,5.,6.]],[[1.,2.,3.],[4.,5.,6.]],[[1.,2.,3.],[4.,5.,6.]]])
# sess = tf.Session()
# mean_none = sess.run(tf.reduce_mean(x,[1,2],keep_dims=True))
#
# print (x)
# print (mean_none)

#sess.close()