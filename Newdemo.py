# Aut# Time: 2019-1-11
# Description:
# Problem:
#       必须输入batch_size大小的数据才能进行训练和预测，在使用中一般是导入一个64*64的矩阵
#       读取大量为数据文件怎么办
#       返回图的结构时名字要相同
#
# Current:
#       在Bolloon上表现还不错，在GTFly序列上表现很差
#       acc_rate可能为nan，因为被除数为0导致的
#
###########################
import numpy as np
import tensorflow as tf
import copy
import math
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import math
import os
import LoadData
import sys
import NewGraph_nl
import time

#
### DEFINE PARAMETER LIST
#

 

#
# CU尺寸：64*64
# GTFly数据分布：前期0:1大概在2:4左右，然后慢慢减少，到了1:29,最后稳定在1:11
# Balloons分布较为稳定，都在2.2:3.8左右
#
YUV = "Balloons"
FILE_TRAIN = "Train_" + YUV + "_Data"
FILE_CROSS = "Cross_" + YUV + "_Data"
FILE_TEST = "Test_" + YUV + "_Data"
# 权值保存路径

pb_file_path = YUV + ".pb"


##############################

def PRINT_PARAMETER(Batch_size,Iteration):
    stdout_backup = sys.stdout
    log_file = open(YUV + "_RunDetail.log", "a")
    sys.stdout = log_file
    print("\n########   Program start   #########")
    print("************************************")
    print("**   Batch_Szie     : ", Batch_size)
    print("**   Iteration      : ", Iteration)
    print("**   alpha          : ", alpha)
    print("**   beta           : ", beta)
    print("**   gamma          : ", gamma)
    print("**   data from file : ", YUV)
    print("************************************")

    log_file.close()
    sys.stdout = stdout_backup
    print("\n************************************")
    print("** Batch_Szie      : ", Batch_size)
    print("** Iteration       : ", Iteration)
    print("** alpha           : ", alpha)
    print("** beta            : ", beta)
    print("** gamma           : ", gamma)
    print("** data from file  : ", YUV)
    print("************************************")


def My_Batch_Norm(inputs):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype=tf.float32)  # input的最后一个维度
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype=tf.float32)
    batch_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    batch_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    return inputs, batch_mean, batch_var, beta, scale

def Reshape_to_32(begin_r,begin_l,last_r,last_l,Mat):
    mat = []
    a = 0
    for i in range(begin_r,last_r):
        mat.append([])
        for j in range(begin_l,last_l):
            mat[a].append(Mat[i][j])
        a += 1
    return np.array(mat)


def Train(graph,Batch_size,Iteration):
    filenames = [("NL测试数据/train_%d" % m +".tfrecords") for m in range(2, 7)]
    print(filenames)
    filenames_cross = ["NL测试数据/train_7.tfrecords"]
    filename_queue = tf.train.string_input_producer(filenames)
    # Tensorflow需要队列流输入数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # tf.parse_single_example为解析器
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "Mat": tf.FixedLenFeature([], tf.string),
                                           "label": tf.FixedLenFeature([], tf.int64),
                                       })

    Mat_16 = tf.decode_raw(features["Mat"], tf.float64)
    Mat_16 = tf.reshape(Mat_16, [16, 16, 1])

    Label_16 = tf.cast(features["label"], tf.float32)

    # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
    #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
    # tf.train.shuffle_batch（）这个函数对读取到的数据进行了batch处理，这样更有利于后续的训练。

    Mat_batch, Label16_batch = tf.train.shuffle_batch \
        ([Mat_16, Label_16], batch_size=Batch_size, capacity=3 * Batch_size, min_after_dequeue=5,
         num_threads=32)
    print("\n--------- finish reading train data-----------")

    filename_queue_cross = tf.train.string_input_producer(filenames_cross)
    reader_cross = tf.TFRecordReader()
    _, serialized_example_cross = reader_cross.read(filename_queue_cross)
    features_cross = tf.parse_single_example(serialized_example_cross,
                                             features={
                                                 "Mat": tf.FixedLenFeature([], tf.string),
                                                 "label": tf.FixedLenFeature([], tf.int64),
                                             })

    Mat_16_cross = tf.decode_raw(features_cross["Mat"], tf.float64)
    Mat_16_cross = tf.reshape(Mat_16_cross, [16, 16, 1])

    Label_16_cross = tf.cast(features_cross["label"], tf.float32)

    Mat_batch_cross, Label16_batch_cross = tf.train.shuffle_batch \
        ([Mat_16_cross, Label_16_cross], batch_size=Batch_size,
         capacity=Batch_size * 3, min_after_dequeue=5,
         num_threads=32)
    # saver = tf.train.Saver()
    # print("Mat_Batttt",Mat_Batch)
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
            acc = 0
            loss = 0
            acc_cross = 0
            loss_cross = 0
            for i in range(Iteration):
                # 这句话的意思是从队列里拿出指定批次的数据
                is_training = True
                mat_16, label_16 = sess.run([Mat_batch, Label16_batch])
                mat_16 = mat_16.reshape(Batch_size, 16, 16, 1)
                # print("Mat_64 is ",Mat_64)
                # print("Label_64 is ",Label_64)
                mat_16 = mat_16.astype('float32')
                label_16 = label_16.astype("float32")
                label_16 = label_16.reshape(Batch_size, 1)

                for b in range(Batch_size):
                    mat_16[b] = LoadData.Normalize(mat_16[b])

                # LOSS_64, acc_rate64, Pre_lable_64, _ = sess.run(
                #     [graph["LOSS"], graph["acc"], graph["Pre"], graph["Train"]],
                #     feed_dict={
                #         graph["inputs_Mat"]: mat_64,
                #         graph["ys_label"]: label_64,
                #         graph['is_training']: is_training,
                #         graph["is_64"]: 1,
                #         graph["is_32"]: 0,
                #         graph["kp"]: 0.5,
                #         graph["ln"]:learning_rate,
                #     })

                # LOSS_32, acc_rate32, Pre_lable_32, _ = sess.run(
                #     [graph["LOSS"], graph["acc"], graph["Pre"], graph["Train"]],
                #     feed_dict={
                #         graph["inputs_Mat"]: mat_32,
                #         graph["ys_label"]: label_32,
                #         graph['is_training']: is_training,
                #         graph["is_64"]: 0,
                #         graph["is_32"]: 1,
                #         graph["kp"]: 0.5,
                #         graph["ln"]: learning_rate,
                #     })
                # print(label_16)
                LOSS_16, acc_rate16, Pre_lable_16, _ = sess.run(
                    [graph["LOSS"], graph["acc"], graph["Pre"], graph["Train"]],
                    feed_dict={
                        graph["inputs_Mat"]: mat_16,
                        graph["ys_label"]: label_16,
                        graph['is_training']: is_training,
                        graph["is_64"]: 0,
                        graph["is_32"]: 0,
                        graph["kp"]: 0.5,
                        graph["ln"]: learning_rate,
                    })

                if i % 100 == 0:
                    is_training = False
                    print("\n------------- Iteration %d --------------" % (i))
                    # # print("LOSS64 is ", LOSS_64)
                    # # print("LOSS32 is ", LOSS_32)
                    # print("LOSS16 is ", LOSS_16)
                    # # print("acc_rate64 is : ", acc_rate64)
                    # # print("acc_rate32 is : ", acc_rate32)
                    # print("acc_rate16 is : ", acc_rate16)

                    # # 把输出写入到文件里
                    # stdout_backup = sys.stdout
                    # log_file = open(YUV + "_RunDetail.log", "a")
                    # sys.stdout = log_file
                    # print("--------------- Iteration %d ---------------" % (i))
                    #
                    # print("LOSS64 is ", LOSS_64)
                    # print("LOSS32 is ", LOSS_32)
                    # print("LOSS16 is ", LOSS_16)
                    # print("acc_rate64 is : ", acc_rate64)
                    # print("acc_rate32 is : ", acc_rate32)
                    # print("acc_rate16 is : ", acc_rate16)
                    mat_cross16, label_cross16 = sess.run([Mat_batch_cross, Label16_batch_cross])
                    mat_cross16 = mat_cross16.reshape(Batch_size, 16, 16, 1)
                    mat_cross16 = mat_cross16.astype("float32")
                    for b in range(Batch_size):
                        mat_cross16[b] = LoadData.Normalize(mat_cross16[b])

                    label_cross16 = label_cross16.astype("float32")
                    label_cross16 = label_cross16.reshape(Batch_size, 1)

                    LOSS_ceoss16, acc_rate_cross16, Pre_lable_cross16 = sess.run(
                        [graph["LOSS"], graph["acc"], graph["Pre"]],
                        feed_dict={
                            graph["inputs_Mat"]: mat_cross16,
                            graph["ys_label"]: label_cross16,
                            graph['is_training']: is_training,
                            graph["is_64"]: 0,
                            graph["is_32"]: 0,
                            graph["kp"]: 1.0,
                        })
                    acc_cross = acc_cross + acc_rate_cross16
                    loss_cross = loss_cross + LOSS_ceoss16
                    loss = loss + LOSS_16
                    acc = acc + acc_rate16
            print("*************train*************")
            print("acc:" + str(acc/60))
            print("loss:" + str(loss/60))
            print("*************cross*************")
            print("acc:" + str(acc_cross / 60))
            print("loss:" + str(loss_cross / 60))
                    # sum16 = 0
                    # a16 = 0
                    # for e in range(Batch_size):
                    #     if label_cross16[e][0] == 1:
                    #         sum16 += 1
                    #         if Pre_lable_cross16[e][0] == 1:
                    #             a16 += 1
                    # if sum16 != 0:
                    #     print("reCall 16 is : ", a16 / sum16)
                    # sum16 = 0
                    # a16 = 0
                    # for e in range(Batch_size):
                    #     if label_cross16[e][0] == 0:
                    #         sum16 += 1
                    #         if Pre_lable_cross16[e][0] == 0:
                    #             a16 += 1
                    # if sum16 != 0:
                    #     print("Call 16 is : ", a16 / sum16)

            if j == 0:
                print("save the model finally:" + str(i))
                var_list = [var for var in tf.global_variables() if "moving" in var.name]
                var_list += tf.trainable_variables()
                saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
                Write_List = ['cond/Merge', 'is_train', 'm_64', 'm_32', 'kp', 'y', 'acc', 'output']
                constant_graph = graph_util.convert_variables_to_constants \
                    (sess, sess.graph_def, Write_List)
                with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                saver.save(sess, "model/Demo-model")
                # saver.save(sess, "./model/Demo-model-one")
                print("save the cross model")

        coord.request_stop()
        print("program ending")
        coord.join(threads)


def End_Program():
    stdout_backup = sys.stdout
    log_file = open(YUV + "_RunDetail.log", "a")
    sys.stdout = log_file
    print("########   Program ending   #########")
    print("\n\n")
    log_file.close()
    sys.stdout = stdout_backup


if __name__ == "__main__":
    demoStart = time.time()
     
    # G0 = NewGraph.Build_Network_16()
    # for r in range(2):
    #     data = [100, 200, 300, 400, 500]
    #     ndata = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # for d in data:
    #     for n in ndata:
    G1 = NewGraph_nl.Build_Network()
    Train(G1,64,2000)
    with tf.Session() as sess:
        initglo = tf.global_variables_initializer()
        initloc = tf.local_variables_initializer()
        sess.run(initglo)
        sess.run(initloc)
        saver = tf.train.import_meta_graph('./model/Demo-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))
        gd = sess.graph.as_graph_def()
        converted_graph_def = graph_util.convert_variables_to_constants(sess, gd,['cond/Merge', 'm_64', 'm_32',
                                                                                     'is_train', 'kp', 'y', 'acc',
                                                                                     'output'])
                                                                                    
        tf.train.write_graph(converted_graph_def, './model','model_nl5.pb', as_text=False)
                                         
            # tf.reset_default_graph()
                    
    End_Program()
    demoEnd = time.time()
    print("Total time cost : ", demoStart - demoEnd)

'''
load_model()
def load_model():
    data = []
    for i in range(64):
        data.append([])
        for j in range(64):
            data[i].append(0)
    data = np.array(data)
    data = data.reshape(1,64,64,1)

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        inputdata0 = graph.get_operation_by_name("input0").outputs[0]
        inputdata1 = graph.get_operation_by_name("input1").outputs[0]
        inputdata2 = graph.get_operation_by_name("input2").outputs[0]

        predict = tf.get_collection("pre64")[0]
        saver = tf.train.import_meta_graph('model/Demo-model.meta')
        saver.restore(sess, tf.train.laCross_checkpoint("model/"))

        print("load model")
        a = sess.run(predict,feed_dict={inputdata0:data,inputdata1:data,inputdata2:data})
        print("predict is ",a[0])
'''
'''
    a = tf.get_default_graph().get_tensor_by_name("Layer/bias:0")
    print("Full connection bias :",sess.run(a))
'''