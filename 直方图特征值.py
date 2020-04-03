import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from collections import Counter


# path_20 = r'E:\pycharm\tensorflow\bubble\20'
# path_20_40 = r'E:\pycharm\tensorflow\bubble\20_40'
# path_40 = r'E:\pycharm\tensorflow\bubble\40'
# path_list_30 = os.listdir(path_20)
# path_list_40 = os.listdir(path_20_40)
# path_list_50 = os.listdir(path_40)
# path_list = [path_list_30, path_list_40, path_list_50]
# path_20 = r'E:\pycharm\tensorflow\data\sort_cut/test_20'
# path_20_40 = r'E:\pycharm\tensorflow\data\sort_cut/test_20_40'
# path_40 = r'E:\pycharm\tensorflow\data\sort_cut/test_40'

# path_20_cut = r'E:\pycharm\tensorflow\data\sort1\train_20'
# path_20_40_cut = r'E:\pycharm\tensorflow\data\sort1\train_20_40'
# path_40_cut = r'E:\pycharm\tensorflow\data\sort1\train_40'
path_20_cut = r'E:\pycharm\tensorflow\data\sort2\40-50-train'
# path_20_40_cut = r'E:\pycharm\tensorflow\data\sort2\test_20_40'
path_40_cut = r'E:\pycharm\tensorflow\data\sort2\40-50-test'
path_list_30 = os.listdir(path_20_cut)
# path_list_40 = os.listdir(path_20_40_cut)
path_list_50 = os.listdir(path_40_cut)
# path_list = [path_list_30, path_list_40, path_list_50]
path_list = [path_list_30, path_list_50]#文件列表


def plot_demo(image):
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    # plt.figure()
    # plt.show("直方图")


def equalHist_demo(image):#均值化
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(image)
    # cv.imshow("equalHist_demo", dst)
    # plot_demo(dst)
    return dst


def blur_demo(image):#均值滤波
    dst = cv.blur(image, (7, 7))
    # cv.imshow("blur_demo", dst)
    # plot_demo(dst)
    return dst


def pre_image(infer_path):#预处理
    src = cv.imread(infer_path)#读取图片
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)#灰度化
    # cv.imshow("gray", gray)
    # plot_demo(gray)
    equ = equalHist_demo(gray)#均值化
    blur = blur_demo(equ)#均值滤波
    return blur


def chara_image(image):#得到特征
    hist = cv.calcHist([image], [0], None, [256], [0, 256])#图片直方图
    Pi = hist/(1716*1438)#像素概念分布
    I = np.arange(256).reshape(256, 1)#像素值
    # sum = np.sum(hist)
    mean = np.sum(Pi * I)#特征1：均值
    var = np.sum(((I - mean)**2)*Pi)#特征2：方差
    std = np.sqrt(var)#标准差
    S = np.sum(((I - mean)**3)*Pi)/(std**3)#特征3：偏差
    K = np.sum(((I - mean)**4)*Pi)/(std**4)#特征4：峰度
    G = np.sum(Pi**2)#特征5：能量
    Pi[Pi == 0] = 1
    H = -np.sum(Pi*np.log(Pi))#特征6：熵
    return mean, var, S, K, G, H


print("--------- Hello Python ---------")

# path_20 = r'E:\pycharm\tensorflow\data\sort1/train_20'
# path_20_40 = r'E:\pycharm\tensorflow\data\sort1/train_20_40'
# path_40 = r'E:\pycharm\tensorflow\data\sort1/train_40'


chara_1 = []#存储文件夹1特征数据
chara_2 = []#存储文件夹2特征数据
# chara_3 = []


def batch_image(path_list):
    global chara_1
    global chara_2
    # global chara_3
    chara = []
    for i in range(2):
        for infer_path in path_list[i]:
            if i == 0:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\40-50-train' + '/' + infer_path
                # sup = [1, 0]
            if i == 1:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\40-50-test'+ '/' + infer_path
                # sup = [0, 1]
            # if i == 2:
            #     infer_path_1 = r'E:\pycharm\tensorflow\data\sort1/test_40' + '/' + infer_path
            #     sup = [0, 0, 1]
            # print(infer_path_1)
            blur_image = pre_image(infer_path_1)#图片预处理
            chara.append(chara_image(blur_image))#特征计算
            # chara.append(sup)
        chara = np.hstack(chara).reshape(len(path_list[i]) , 6)
        print(chara.shape)
        if i == 0:
            chara_1 = chara.copy()
        if i == 1:
            chara_2 = chara.copy()
        # if i == 2:
        #     chara_3 = chara.copy()
        chara = []


batch_image(path_list)#得到特征值存储在chara_1和chara_2中
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_1.txt", chara_1)
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_2.txt", chara_2)
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_3.txt", chara_3)
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_40_50_whole.txt", chara_1)#保存数据
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_40_50_whole.txt", chara_2)
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt", chara_3)
print(chara_1[0])
print(chara_2[0])
# print(chara_3[0])

train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_40_50_whole.txt")
train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_40_50_whole.txt")
# train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt")
print(train_chara_1.shape)
print(train_chara_2.shape)
# print(train_chara_3.shape)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
