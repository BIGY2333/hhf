import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist

# 设置权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# 设置阈值函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 设置卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = "SAME")

# 设置池化层
def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME")

def SE_block(x,ratio):

    shape = x.get_shape().as_list()
    channel_out = shape[3]
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation"):
        # 第一层，全局平均池化层
        squeeze = tf.nn.avg_pool(x,[1,shape[1],shape[2],1],[1,shape[1],shape[2],1],padding = "SAME")
        # 第二层，全连接层
        w_excitation1 = weight_variable([1,1,channel_out,channel_out/ratio])
        b_excitation1 = bias_variable([channel_out/ratio])
        excitation1 = conv2d(squeeze,w_excitation1) + b_excitation1
        excitation1_output = tf.nn.relu(excitation1)
        # 第三层，全连接层
        w_excitation2 = weight_variable([1, 1, channel_out / ratio, channel_out])
        b_excitation2 = bias_variable([channel_out])
        excitation2 = conv2d(excitation1_output, w_excitation2) + b_excitation2
        excitation2_output = tf.nn.sigmoid(excitation2)
        # 第四层，点乘
        excitation_output = tf.reshape(excitation2_output,[-1,1,1,channel_out])
        h_output = excitation_output * x

    return h_output

def graph_build():
    x_image = tf.placeholder(tf.float32, (None, 32, 32, 1), name='X')
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h1_conv = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h1_conv = SE_block(h1_conv, 4)
    # 配置第一层池化层
    h1_pool = pool(h1_conv)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h2_conv = tf.nn.relu(conv2d(h1_pool, W_conv2) + b_conv2)
    h2_conv = SE_block(h2_conv, 4)
    # 配置第二层池化层
    h2_pool = pool(h2_conv)

    # 配置全连接层1
    W_fc1 = weight_variable([8 * 8 * 64, 256])
    b_fc1 = bias_variable([256])
    h2_pool_flat = tf.reshape(h2_pool, shape=[-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)

    # 配置dropout层
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    h3_pool_flat = tf.reshape(h_fc1_drop, shape=[-1, 8 * 8 * 64])
    h_fc2 = tf.nn.relu(tf.matmul(h3_pool_flat, W_fc2) + b_fc2)

    # 配置全连接层2
    W_fc3 = weight_variable([10, 1])
    b_fc3 = bias_variable([1])
    y_predict = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # 接受y_label
    y_label = tf.placeholder(tf.float32, [None, 1])
    mnist = get_data()

    # 交叉熵
    cross_entropy = -tf.reduce_sum(y_label * tf.log(y_predict))
    # 梯度下降法
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    # 求准确率
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

