import numpy as np
import tensorflow as tf

Path = "D:/文档/工程/二维熵/PARK/QP30/"


def LoadData():
    # Newspaper 31488
    # Street 13300
    # GTFly 13300
    Name = ['Balloons']
    #Name = ['Balloons']0
    width = [32]
    num = ['1']
    for l in num:
        tfrecords_filename = "D:/文档/工程/HEVC-神经网络/Data/NL测试数据/train_"+l+".tfrecords"
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for j in width:
            for i in Name:
                Mat_s = np.loadtxt(Path + i + "/" + i + str(j) + '-Split_'+l+'.txt')
                Mat_ns = np.loadtxt(Path + i + "/" + i + str(j) + '-Nonsplit_'+l+'.txt')
                print(Mat_s.shape[0] // j)
                Mat_s = Mat_s.reshape(Mat_s.shape[0] // j, j, j)
                Mat_ns = Mat_ns.reshape(Mat_ns.shape[0] // j, j, j)
                for k in range(1000):
                    for m in [0, 1]:
                        if m == 1:
                            Mat_str = Mat_s[k].tostring()
                            Lab = []
                            Lab.append(1)
                            Lab = np.array(Lab)
                            example = tf.train.Example(features=tf.train.Features(
                                feature={
                                    "Mat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[Mat_str])),
                                    "label": tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[[1,0])),
                                }))
                            writer.write(example.SerializeToString())
                        elif m == 0:
                            Mat_nstr = Mat_ns[k].tostring()
                            Lab = []
                            Lab.append(0)
                            Lab = np.array(Lab)
                            example = tf.train.Example(features=tf.train.Features(
                                feature={
                                    "Mat": tf.train.Feature(
                                        bytes_list=tf.train.BytesList(value=[Mat_nstr])),
                                    "label": tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[[0,1])),
                                }))
                            writer.write(example.SerializeToString())
                '''    
                elif i == 'Balloons' or i == 'Kendo' or i == 'Newspaper':
                    for k in range(5000):
                        for m in [0, 1]:
                            if m == 0 and k < 5000 - d:
                                Mat_str = Mat_s[k].tostring()
                                Lab = []
                                Lab.append(1)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[Mat_str])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                            elif m == 1:
                                Mat_nstr = Mat_ns[k].tostring()
                                Lab = []
                                Lab.append(0)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(value=[Mat_nstr])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                elif i == 'Dancer':
                    for k in range(8000):
                        for m in [0, 1]:
                            if m == 0 and k < 8000:
                                Mat_str = Mat_s[k].tostring()
                                Lab = []
                                Lab.append(1)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[Mat_str])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                            elif m == 1:
                                Mat_nstr = Mat_ns[k].tostring()
                                Lab = []
                                Lab.append(0)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(value=[Mat_nstr])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                elif i == 'GT_fly':
                    for k in range(8000):
                        for m in [0, 1]:
                            if m == 0 and k < 8000:
                                Mat_str = Mat_s[k].tostring()
                                Lab = []
                                Lab.append(1)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[Mat_str])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                            elif m == 1:
                                Mat_nstr = Mat_ns[k].tostring()
                                Lab = []
                                Lab.append(0)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(value=[Mat_nstr])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                else:
                    for k in range(5000):
                        for m in [0, 1]:
                            if m == 0 and k < 5000:
                                Mat_str = Mat_s[k].tostring()
                                Lab = []
                                Lab.append(1)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[Mat_str])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
                            elif m == 1:
                                Mat_nstr = Mat_ns[k].tostring()
                                Lab = []
                                Lab.append(0)
                                Lab = np.array(Lab)
                                example = tf.train.Example(features=tf.train.Features(
                                    feature={
                                        "Mat": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(value=[Mat_nstr])),
                                        "label": tf.train.Feature(
                                            int64_list=tf.train.Int64List(value=[Lab[0]])),
                                    }))
                                writer.write(example.SerializeToString())
        '''
        writer.close()


def Read_TFRData():
    tfrecords_filename = "D:/文档/工程/HEVC-神经网络/Data/Data16/train_1.tfrecords"
    filename_queue = tf.train.string_input_producer([tfrecords_filename], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "Mat": tf.FixedLenFeature([], tf.string),
                                           "label": tf.FixedLenFeature([], tf.int64),
                                       })
    Mat = tf.decode_raw(features["Mat"], tf.int64)
    print(type(Mat))
    Mat_64 = tf.reshape(Mat, [64, 64, 1])
    Mat_64 = tf.cast(Mat_64, tf.float32) * (1.0 / 255)
    Mat_64 = tf.image.per_image_standardization(Mat_64)
    Label_32 = tf.decode_raw(features["label_32"], tf.int64)
    Label_32 = tf.reshape(Label_32, [4])
    print("Label_32 shape", Label_32.shape)
    Label_16 = tf.decode_raw(features["label_16"], tf.int64)
    Label_16 = tf.reshape(Label_16, [16])
    print("Label_16 shape is ", Label_16.shape)
    Label_64 = tf.cast(features["label_64"], tf.int64)
    Mat_batch, Label64_batch, Label32_batch, Label16_Batch = tf.train.shuffle_batch \
        ([Mat_64, Label_64, Label_32, Label_16], batch_size=3, capacity=10, min_after_dequeue=5, num_threads=1)
    print("-----------read data ending------------")
    return Mat_batch, Label64_batch, Label32_batch, Label16_Batch


if __name__ == "__main__":

    LoadData()

    # filenames = [("D:/文档/工程/HEVC-神经网络/Data/Data16/train_%d.tfrecords" % j) for j in range(1, 7)]
    # filename_queue = tf.train.string_input_producer(filenames)
    # # Tensorflow需要队列流输入数据
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # # tf.parse_single_example为解析器
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        "Mat": tf.FixedLenFeature([], tf.string),
    #                                        "label": tf.FixedLenFeature([], tf.int64),
    #                                    })
    #
    # Mat = tf.decode_raw(features["Mat"], tf.float64)
    # Mat = tf.reshape(Mat, [16, 16, 1])
    # Label = tf.cast(features["label"], tf.int64)
    #
    # # Mat_batch,Label64_batch,Label32_batch,Label16_Batch = tf.train.batch\
    # #                ([Mat_64, Label_64,Label_32,Label_16],batch_size= Batch_size,capacity=300,num_threads= 1)
    # # tf.train.shuffle_batch（）这个函数对读取到的数据进行了batch处理，这样更有利于后续的训练。
    #
    # Mat_batch, Label_batch = tf.train.shuffle_batch \
    #     ([Mat, Label], batch_size=16, capacity=300, min_after_dequeue=5,
    #      num_threads=1)
    # print("\n--------- finish reading train data-----------")
    # # saver = tf.train.Saver()
    # # print("Mat_Batttt",Mat_Batch)
    # with tf.Session() as sess:
    #     # 开启协调器
    #     coord = tf.train.Coordinator()
    #     # 使用start_queue_runners 启动队列填充
    #     # 一定要开启队列，要不然是空的
    #     threads = tf.train.start_queue_runners(sess, coord)
    #     initglo = tf.global_variables_initializer()
    #     initloc = tf.local_variables_initializer()
    #     sess.run(initglo)
    #     sess.run(initloc)
    #     tf.local_variables_initializer().run()
    #     epoch = 0
    #     for i in range(10):
    #         # 这句话的意思是从队列里拿出指定批次的数据
    #         is_training = True
    #         mat_16, label_16 = sess.run([Mat_batch, Label_batch])
    #         # print("Mat_64 is ",Mat_64)
    #         # print("Label_64 is ",Label_64)
    #         mat_16 = mat_16.reshape(16, 16, 16, 1)
    #         print(mat_16)
    #         print(label_16)
