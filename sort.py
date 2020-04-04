import os
import random

import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_10_20_whole_54.txt")
train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_20_30_whole_54.txt")
train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_30_40_whole_54.txt")
train_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_40_50_whole_54.txt")
train_chara_5 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_54.txt")
test_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_10_20_whole_54.txt")
test_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_20_30_whole_54.txt")
test_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_30_40_whole_54.txt")
test_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_40_50_whole_54.txt")
test_chara_5 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_54.txt")
vary_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\20-30_all_54.txt")
vary_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\30-40_all_54.txt")
vary_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\40-50_all_54.txt")
# train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_10_20_whole_45.txt")
# train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_20_30_whole_45.txt")
# train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_30_40_whole_45.txt")
# train_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_40_50_whole_45.txt")
# train_chara_5 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_45.txt")
# test_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_10_20_whole_45.txt")
# test_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_20_30_whole_45.txt")
# test_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_30_40_whole_45.txt")
# test_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_40_50_whole_45.txt")
# test_chara_5 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_45.txt")
# vary_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\20-30_all_45.txt")
# vary_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\30-40_all_45.txt")
# vary_chara_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\check\40-50_all_45.txt")
spit1 = 49
spit2 = 4
many = 2
train_chara_1_spit = np.hsplit(train_chara_1, np.array([spit2, spit1]))
train_chara_2_spit = np.hsplit(train_chara_2, np.array([spit2, spit1]))
train_chara_3_spit = np.hsplit(train_chara_3, np.array([spit2, spit1]))
train_chara_4_spit = np.hsplit(train_chara_4, np.array([spit2, spit1]))
train_chara_5_spit = np.hsplit(train_chara_5, np.array([spit2, spit1]))
test_chara_1_spit = np.hsplit(test_chara_1, np.array([spit2, spit1]))
test_chara_2_spit = np.hsplit(test_chara_2, np.array([spit2, spit1]))
test_chara_3_spit = np.hsplit(test_chara_3, np.array([spit2, spit1]))
test_chara_4_spit = np.hsplit(test_chara_4, np.array([spit2, spit1]))
test_chara_5_spit = np.hsplit(test_chara_5, np.array([spit2, spit1]))
# test_chara_1_spit = np.hsplit(test_chara_1, np.array([49]))
vary_chara_2_spit = np.hsplit(vary_chara_2, np.array([spit2, spit1]))
vary_chara_3_spit = np.hsplit(vary_chara_3, np.array([spit2, spit1]))
vary_chara_4_spit = np.hsplit(vary_chara_4, np.array([spit2, spit1]))
# test_chara_5_spit = np.hsplit(test_chara_5, np.array([49]))
scaled_data_1_train = preprocessing. scale(train_chara_1_spit[many-1])
scaled_data_2_train = preprocessing. scale(train_chara_2_spit[many-1])
scaled_data_3_train = preprocessing. scale(train_chara_3_spit[many-1])
scaled_data_4_train = preprocessing. scale(train_chara_4_spit[many-1])
scaled_data_5_train = preprocessing. scale(train_chara_5_spit[many-1])
scaled_data_1_test = preprocessing. scale(test_chara_1_spit[many-1])
scaled_data_2_test = preprocessing. scale(test_chara_2_spit[many-1])
scaled_data_3_test = preprocessing. scale(test_chara_3_spit[many-1])
scaled_data_4_test = preprocessing. scale(test_chara_4_spit[many-1])
scaled_data_5_test = preprocessing. scale(test_chara_5_spit[many-1])
# scaled_data_1_test = preprocessing. scale(test_chara_1_spit[0])
scaled_data_2_vary = preprocessing. scale(vary_chara_2_spit[many-1])
scaled_data_3_vary = preprocessing. scale(vary_chara_3_spit[many-1])
scaled_data_4_vary = preprocessing. scale(vary_chara_4_spit[many-1])
# scaled_data_5_test = preprocessing. scale(test_chara_5_spit[0])
print(scaled_data_1_train.shape)
print(scaled_data_2_train.shape)
print(scaled_data_3_train.shape)
print(scaled_data_4_train.shape)
print(scaled_data_5_train.shape)
print(scaled_data_1_test.shape)
print(scaled_data_2_test.shape)
print(scaled_data_3_test.shape)
print(scaled_data_4_test.shape)
print(scaled_data_5_test.shape)
# print(scaled_data_1_test.shape)
print(scaled_data_2_vary.shape)
print(scaled_data_3_vary.shape)
print(scaled_data_4_vary.shape)
# print(scaled_data_5_test.shape)
train_data = np.concatenate((scaled_data_1_train,scaled_data_2_train,scaled_data_3_train,scaled_data_4_train,scaled_data_5_train), axis=0)
test_data = np.concatenate((scaled_data_1_test,scaled_data_2_test,scaled_data_3_test,scaled_data_4_test,scaled_data_5_test), axis=0)
vary_data = np.concatenate((scaled_data_2_vary,scaled_data_3_vary,scaled_data_4_vary), axis=0)
train_data_lable = np.concatenate((train_chara_1_spit[many],train_chara_2_spit[many],train_chara_3_spit[many],train_chara_4_spit[many],train_chara_5_spit[many]), axis=0)
test_data_lable = np.concatenate((test_chara_1_spit[many],test_chara_2_spit[many],test_chara_3_spit[many],test_chara_4_spit[many],test_chara_5_spit[many]), axis=0)
vary_data_lable = np.concatenate((vary_chara_2_spit[many],vary_chara_3_spit[many],vary_chara_4_spit[many]), axis=0)
pca = PCA()
pca.fit(train_data)
pca_data_train = pca.transform(train_data)
pca_data_test = pca.transform(test_data)
pca_data_vary = pca.transform(vary_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
print(per_var)
# data_image.append(train_chara_2_spit[0][0])
# data_label.append(train_chara_2_spit[1][0])
# print(data_image)
# print(data_label)
batch_size = 100
learning_rate = 0.02
learning_rate_decay = 0.99
max_steps = 1500
trait = 45
trait_pca = 20

def hidden_layer_pca(input_tensor,keep_prob, regularizer,avg_class,resuse):

    Full_connection1_weights = tf.Variable(tf.random_normal([trait_pca, 200]))
    #if regularizer != None:
    tf.add_to_collection("losses_pca", regularizer(Full_connection1_weights))
    Full_connection1_biases = tf.Variable(tf.random_normal([200]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class ==None:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, Full_connection1_weights) + \
                                                               Full_connection1_biases)
    else:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, avg_class.average(Full_connection1_weights))
                                               + avg_class.average(Full_connection1_biases))
    h_fc1_drop = tf.nn.dropout(Full_1, keep_prob)



    Full_connection2_weights = tf.Variable(tf.random_normal([200, 100]))
    # if regularizer != None:
    tf.add_to_collection("losses_pca", regularizer(Full_connection2_weights))
    Full_connection2_biases = tf.Variable(tf.random_normal([100]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, Full_connection2_weights) + \
                            Full_connection2_biases)
    else:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, avg_class.average(Full_connection2_weights))
                            + avg_class.average(Full_connection2_biases))
    h_fc2_drop = tf.nn.dropout(Full_2, keep_prob)

    # Full_connection3_weights = tf.Variable(tf.random_normal([100, 50]))
    # # if regularizer != None:
    # tf.add_to_collection("losses", regularizer(Full_connection3_weights))
    # Full_connection3_biases = tf.Variable(tf.random_normal([50]))
    # # variables_averages_op = avg_class.apply(tf.trainable_variables())
    # if avg_class == None:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, Full_connection3_weights) + \
    #                         Full_connection3_biases)
    # else:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, avg_class.average(Full_connection3_weights))
    #                         + avg_class.average(Full_connection3_biases))
    # h_fc3_drop = tf.nn.dropout(Full_3, keep_prob)


    Full_connection4_weights = tf.Variable(tf.random_normal([100, 5]))
    #if regularizer != None:
    tf.add_to_collection("losses_pca", regularizer(Full_connection4_weights))
    Full_connection4_biases = tf.Variable(tf.random_normal([5]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        result = tf.matmul(h_fc2_drop, Full_connection4_weights) + Full_connection4_biases
    else:
        result = tf.matmul(h_fc2_drop, avg_class.average(Full_connection4_weights)) + \
                                              avg_class.average(Full_connection4_biases)
    return result
class RBF:
    # 初始化学习率、学习步数
    def __init__(self):
        pass

    # 高斯核函数(c为中心，s为标准差)
    def kernel(self, x, c, s, hidden_size, feature):
        x1 = tf.tile(x, [1, hidden_size])  # 将x水平复制 hidden次
        x2 = tf.reshape(x1, [-1, hidden_size, feature])
        dist = tf.reduce_sum((x2 - c) ** 2, 2)
        return tf.exp(-dist / (2 * s ** 2))

    # 训练RBF神经网络
    def rbf(self, x, feature, hidden_size):
        # feature = np.shape(x)[1]  # 输入值的特征数
        c = tf.Variable(tf.random_normal([hidden_size, feature]))
        s = tf.Variable(tf.random_normal([hidden_size]))
        z = self.kernel(x, c, s, hidden_size, feature)
        return z


def hidden_layer_rbf(input_tensor,keep_prob, regularizer,avg_class,resuse):

    Full_connection1_weights = tf.Variable(tf.random_normal([200, 200]))
    #if regularizer != None:
    tf.add_to_collection("losses_rbf", regularizer(Full_connection1_weights))
    Full_connection1_biases = tf.Variable(tf.random_normal([200]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class ==None:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, Full_connection1_weights) + \
                                                               Full_connection1_biases)
    else:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, avg_class.average(Full_connection1_weights))
                                               + avg_class.average(Full_connection1_biases))
    h_fc1_drop = tf.nn.dropout(Full_1, keep_prob)



    Full_connection2_weights = tf.Variable(tf.random_normal([200, 100]))
    # if regularizer != None:
    tf.add_to_collection("losses_rbf", regularizer(Full_connection2_weights))
    Full_connection2_biases = tf.Variable(tf.random_normal([100]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, Full_connection2_weights) + \
                            Full_connection2_biases)
    else:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, avg_class.average(Full_connection2_weights))
                            + avg_class.average(Full_connection2_biases))
    h_fc2_drop = tf.nn.dropout(Full_2, keep_prob)

    # Full_connection3_weights = tf.Variable(tf.random_normal([100, 50]))
    # # if regularizer != None:
    # tf.add_to_collection("losses", regularizer(Full_connection3_weights))
    # Full_connection3_biases = tf.Variable(tf.random_normal([50]))
    # # variables_averages_op = avg_class.apply(tf.trainable_variables())
    # if avg_class == None:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, Full_connection3_weights) + \
    #                         Full_connection3_biases)
    # else:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, avg_class.average(Full_connection3_weights))
    #                         + avg_class.average(Full_connection3_biases))
    # h_fc3_drop = tf.nn.dropout(Full_3, keep_prob)


    Full_connection4_weights = tf.Variable(tf.random_normal([100, 5]))
    #if regularizer != None:
    tf.add_to_collection("losses_rbf", regularizer(Full_connection4_weights))
    Full_connection4_biases = tf.Variable(tf.random_normal([5]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        result = tf.matmul(h_fc2_drop, Full_connection4_weights) + Full_connection4_biases
    else:
        result = tf.matmul(h_fc2_drop, avg_class.average(Full_connection4_weights)) + \
                                              avg_class.average(Full_connection4_biases)
    return result

def hidden_layer(input_tensor,keep_prob, regularizer,avg_class,resuse):

    Full_connection1_weights = tf.Variable(tf.random_normal([trait, 200]))
    #if regularizer != None:
    tf.add_to_collection("losses", regularizer(Full_connection1_weights))
    Full_connection1_biases = tf.Variable(tf.random_normal([200]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class ==None:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, Full_connection1_weights) + \
                                                               Full_connection1_biases)
    else:
        Full_1 = tf.nn.tanh(tf.matmul(input_tensor, avg_class.average(Full_connection1_weights))
                                               + avg_class.average(Full_connection1_biases))
    h_fc1_drop = tf.nn.dropout(Full_1, keep_prob)



    Full_connection2_weights = tf.Variable(tf.random_normal([200, 100]))
    # if regularizer != None:
    tf.add_to_collection("losses", regularizer(Full_connection2_weights))
    Full_connection2_biases = tf.Variable(tf.random_normal([100]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, Full_connection2_weights) + \
                            Full_connection2_biases)
    else:
        Full_2 = tf.nn.tanh(tf.matmul(h_fc1_drop, avg_class.average(Full_connection2_weights))
                            + avg_class.average(Full_connection2_biases))
    h_fc2_drop = tf.nn.dropout(Full_2, keep_prob)

    # Full_connection3_weights = tf.Variable(tf.random_normal([100, 50]))
    # # if regularizer != None:
    # tf.add_to_collection("losses", regularizer(Full_connection3_weights))
    # Full_connection3_biases = tf.Variable(tf.random_normal([50]))
    # # variables_averages_op = avg_class.apply(tf.trainable_variables())
    # if avg_class == None:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, Full_connection3_weights) + \
    #                         Full_connection3_biases)
    # else:
    #     Full_3 = tf.nn.tanh(tf.matmul(h_fc2_drop, avg_class.average(Full_connection3_weights))
    #                         + avg_class.average(Full_connection3_biases))
    # h_fc3_drop = tf.nn.dropout(Full_3, keep_prob)


    Full_connection4_weights = tf.Variable(tf.random_normal([100, 5]))
    #if regularizer != None:
    tf.add_to_collection("losses", regularizer(Full_connection4_weights))
    Full_connection4_biases = tf.Variable(tf.random_normal([5]))
    # variables_averages_op = avg_class.apply(tf.trainable_variables())
    if avg_class == None:
        result = tf.matmul(h_fc2_drop, Full_connection4_weights) + Full_connection4_biases
    else:
        result = tf.matmul(h_fc2_drop, avg_class.average(Full_connection4_weights)) + \
                                              avg_class.average(Full_connection4_biases)
    return result


x_pca = tf.placeholder(tf.float32, [batch_size ,trait_pca],name="x-input_pca")
y_pca = tf.placeholder(tf.float32, [None, 5], name="y-input_pca")

x_rbf = tf.placeholder(tf.float32, [batch_size ,trait],name="x-input_rbf")
y_rbf = tf.placeholder(tf.float32, [None, 5], name="y-input_rbf")

x_ = tf.placeholder(tf.float32, [batch_size ,trait],name="x-input")
y_ = tf.placeholder(tf.float32, [None, 5], name="y-input")

keep_prob = tf.placeholder(tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(0.0001)



# training_step = tf.Variable(0, trainable=False)
# variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
# variables_averages_op = variable_averages.apply(tf.trainable_variables())
rbf = RBF()
z_ = rbf.rbf(x_rbf, trait, 200)

y1_pca = hidden_layer_pca(x_pca,keep_prob, regularizer,avg_class=None,resuse=False)
y1_rbf= hidden_layer_rbf(z_,keep_prob, regularizer,avg_class=None,resuse=False)
y1 = hidden_layer(x_,keep_prob, regularizer,avg_class=None,resuse=False)


cross_entropy_pca = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y1_pca, labels=tf.argmax(y_pca, 1))
cross_entropy_mean_pca = tf.reduce_mean(cross_entropy_pca)
cross_entropy_rbf = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y1_rbf, labels=tf.argmax(y_rbf, 1))
cross_entropy_mean_rbf = tf.reduce_mean(cross_entropy_rbf)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y1, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss_pca = cross_entropy_mean_pca + tf.add_n(tf.get_collection('losses_pca'))
loss_rbf = cross_entropy_mean_rbf + tf.add_n(tf.get_collection('losses_rbf'))
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# learning_rate = tf.train.exponential_decay(learning_rate,
#                                  training_step, 3000/batch_size ,
#                                  learning_rate_decay, staircase=True)
#
# train_step = tf.train.GradientDescentOptimizer(learning_rate). \
#     minimize(loss, global_step=training_step)
# learning_rate = tf.train.exponential_decay(learning_rate,
#                                  training_step, 3000/batch_size ,
#                                  learning_rate_decay, staircase=True)
train_step_pca = tf.train.AdamOptimizer(0.001).minimize(loss_pca)
train_step_rbf = tf.train.AdamOptimizer(0.001).minimize(loss_rbf)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
equ = tf.equal(tf.arg_max(y1_pca,1),tf.argmax(y1,1))
equ_1 = tf.reshape(equ, shape=[100, 1])
equ_2 = tf.tile(tf.to_float(equ_1),[1,5])
# print(equ_1.shape)
# tf.tile(a, [2, 3])
y_result =tf.multiply(equ_2,y1_pca)+tf.multiply((equ_2*(-1))+1,y1_rbf)



# with tf.control_dependencies([train_step, variables_averages_op]):
#     train_op = tf.no_op(name='train')
crorent_predicition = tf.equal(tf.arg_max(y_result,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))
saver = tf.train.Saver()
data_image_train_pca = []
data_label_train_pca = []
data_image_train_rbf = []
data_label_train_rbf = []
data_image_train_bp = []
data_label_train_bp = []

data_image_test_pca = []
# data_label_test_pca = []
data_image_test_rbf = []
# data_label_test_rbf = []
data_image_test_bp = []
data_label_test = []
data_image_vary_pca = []
# data_label_vary_pca = []
data_image_vary_rbf = []
# data_label_vary_rbf = []
data_image_vary_bp = []
data_label_vary = []


data_image_pca = []
data_label_pca = []
data_image_rbf = []
data_label_rbf = []
data_image = []
data_label = []
k1 = 0
k2 = 0
k3 = 0
k4 = 0
k5 = 0
ave = []
with tf.Session() as sess:
    # saver.restore(sess, r"E:\pycharm\tensorflow\data\sort2\1.rbf/1.ckpt")

    tf.global_variables_initializer().run()
    for p in range(150000):
        for j in range(100):
            i1 = np.random.randint(0, pca_data_train.shape[0], [1])  # 从哪个文件夹里抽
            data_image_train_pca.append(pca_data_train[i1])
            data_label_train_pca.append(train_data_lable[i1])

            i1 = np.random.randint(0, train_data.shape[0], [1])  # 从哪个文件夹里抽
            data_image_train_rbf.append(train_data[i1])
            data_label_train_rbf.append(train_data_lable[i1])

            i1 = np.random.randint(0, train_data.shape[0], [1])  # 从哪个文件夹里抽
            data_image_train_bp.append(train_data[i1])
            data_label_train_bp.append(train_data_lable[i1])
        data_image_pca = np.vstack(data_image_train_pca).reshape(1, 100, trait)
        data_label_pca = np.vstack(data_label_train_pca).reshape(1, 100, 5)
        data_image_rbf = np.vstack(data_image_train_rbf).reshape(1, 100, trait)
        data_label_rbf = np.vstack(data_label_train_rbf).reshape(1, 100, 5)
        data_image = np.vstack(data_image_train_bp).reshape(1, 100, trait)
        data_label = np.vstack(data_label_train_bp).reshape(1, 100, 5)
        sess.run(train_step_pca, feed_dict={x_pca: data_image_pca[0][:,0:trait_pca], y_pca: data_label_pca[0], keep_prob:0.5})
        sess.run(train_step_rbf, feed_dict={x_rbf: data_image_rbf[0], y_rbf: data_label_rbf[0], keep_prob:0.5})
        sess.run(train_step, feed_dict={x_: data_image[0], y_: data_label[0], keep_prob:0.5})
        data_image_pca = []
        data_label_pca = []
        data_image_rbf = []
        data_label_rbf = []
        data_image = []
        data_label = []
        if p % 1000 == 0:
            i1 = random.sample(range(0,pca_data_test.shape[0]),100)# 抽出哪几个
            # print(i1)
            for i in i1:
                    # i1 = np.random.randint(0, pca_data_test.shape[0], [1])  # 从哪个文件夹里抽
                data_image_test_pca.append(pca_data_test[i])
                data_image_test_rbf.append(test_data[i])
                data_image_test_bp.append(test_data[i])
                data_label_test.append(test_data_lable[i])
            data_image_pca = np.vstack(data_image_test_pca).reshape(1, 100, trait)
            data_image_rbf = np.vstack(data_image_test_rbf).reshape(1, 100, trait)
            data_image = np.vstack(data_image_test_bp).reshape(1, 100, trait)
            data_label = np.vstack(data_label_test).reshape(1, 100, 5)
            mse = sess.run(accuracy, feed_dict={x_pca: data_image_pca[0][:,0:trait_pca],x_rbf: data_image_rbf[0],x_: data_image[0], y_: data_label[0], keep_prob:0.5})
            print(p, mse)
            data_image_pca = []
            data_image_rbf = []
            data_image = []
            data_label = []

        if p % 1000 == 0:
            i1 = random.sample(range(0, pca_data_vary.shape[0]), 100)  # 抽出哪几个
            # print(i1)
            for i in i1:
                # i1 = np.random.randint(0, pca_data_test.shape[0], [1])  # 从哪个文件夹里抽
                data_image_vary_pca.append(pca_data_vary[i])
                data_image_vary_rbf.append(vary_data[i])
                data_image_vary_bp.append(vary_data[i])
                data_label_vary.append(vary_data_lable[i])
            data_image_pca = np.vstack(data_image_vary_pca).reshape(1, 100, trait)
            data_image_rbf = np.vstack(data_image_vary_rbf).reshape(1, 100, trait)
            data_image = np.vstack(data_image_vary_bp).reshape(1, 100, trait)
            data_label = np.vstack(data_label_vary).reshape(1, 100, 5)
            mse = sess.run(accuracy,
                           feed_dict={x_pca: data_image_pca[0][:, 0:trait_pca], x_rbf: data_image_rbf[0], x_: data_image[0],
                                      y_: data_label[0], keep_prob: 0.5})
            print(p, mse)
            data_image_pca = []
            data_image_rbf = []
            data_image = []
            data_label = []
        # acc = 100 * sess.run(accuracy, feed_dict={x: data_image[0], y_: data_label[0], keep_prob: 0.5})
        # print("In %d circle, the accuracy of 100 picture is %g%%" % (p, acc))
        # ave.append(acc)
        data_image_train_pca = []
        data_label_train_pca = []
        data_image_train_rbf = []
        data_label_train_rbf = []
        data_image_train_bp = []
        data_label_train_bp = []

        data_image_test_pca = []
        # data_label_test_pca = []
        data_image_test_rbf = []
        # data_label_test_rbf = []
        data_image_test_bp = []
        data_label_test = []
        data_image_vary_pca = []
        # data_label_vary_pca = []
        data_image_vary_rbf = []
        # data_label_vary_rbf = []
        data_image_vary_bp = []
        data_label_vary = []

        # print(sess.run(y_pre, feed_dict={x: b[0]}))
    saver.save(sess, r"E:\pycharm\tensorflow\data\sort2\3.bag54/1.ckpt")
    # print((sum(ave) - max(ave) - min(ave)) / (len(ave) - 2))


