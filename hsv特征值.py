#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2 as cv
import math
import numpy as np
import os
from matplotlib import pyplot as plt

from numpy import *
# 定义最大灰度级数
gray_level = 16

path_20_cut = r'E:\pycharm\tensorflow\data\sort2\50-60-train'
# path_20_40_cut = r'E:\pycharm\tensorflow\data\sort2\test_20_40'
path_40_cut = r'E:\pycharm\tensorflow\data\sort2\50-60-test'
path_list_30 = os.listdir(path_20_cut)
# path_list_40 = os.listdir(path_20_40_cut)
path_list_50 = os.listdir(path_40_cut)
# path_list = [path_list_30, path_list_40, path_list_50]
path_list = [path_list_30, path_list_50]#文件列表
#
#
# def maxGrayLevel(img):
#     max_gray_level = 0
#     (height, width) = img.shape
#     print(height, width)
#     for y in range(height):
#         for x in range(width):
#             if img[y][x] > max_gray_level:
#                 max_gray_level = img[y][x]
#     return max_gray_level + 1
#
#


def plot_demo(image):
    # plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    # plt.figure()


def equalHist_demo(image):#均衡化
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(image)
    # cv.imshow("equalHist_demo", dst)
    # plot_demo(dst)
    return dst


def blur_demo(image):#领域均值滤波
    dst = cv.blur(image, (7, 7))
    # cv.imshow("blur_demo", dst)
    # plot_demo(dst)
    return dst


def pre_image(infer_path):
    src = cv.imread(infer_path)#读取图像

    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

    # cv.imshow("src1", src)
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)#BGR转HSV格式
    # cv.imshow("hsv", hsv)
    value = hsv[:, :, 2]#提取选择v值
    # plt.figure(1)
    # plot_demo(value)
    # cv.imshow("value", value)
    # cv.imshow("gray", gray)
    # plot_demo(gray)
    equ = equalHist_demo(value)#均衡化
    # plt.figure(2)
    # plot_demo(equ)
    # cv.imshow("equ", src)
    image = blur_demo(equ)#领域平均滤波
    # plt.figure(3)
    # plot_demo(image)
    # cv.imshow("image", image)
    return image



# def getGlcm(input, d_x, d_y):
#     srcdata = input.copy()
#     ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
#     # ret = np.zeros([16, 16])
#
#     (height, width) = input.shape
#
#     max_gray_level = 256
#
#     # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
#     # if max_gray_level > gray_level:
#     #     for j in range(height):
#     #         for i in range(width):
#     #             srcdata[j][i] = (srcdata[j][i] * gray_level / max_gray_level)
#
#     srcdata = (srcdata.astype(float) * gray_level / max_gray_level).astype(int)
#     if d_x>=0:
#         for j in range(height - d_y):
#             for i in range(width - d_x):
#                 rows = srcdata[j][i]
#                 cols = srcdata[j + d_y][i + d_x]
#                 ret[rows][cols] += 1.0
#     if d_x<0:
#         for j in range(height - d_y):
#             for i in np.arange(-d_x, width):
#                 rows = srcdata[j][i]
#                 cols = srcdata[j + d_y][i + d_x]
#                 ret[rows][cols] += 1.0
#
#     for i in range(gray_level):
#         for j in range(gray_level):
#             ret[i][j] /= float((height - d_y) * (width - abs(d_x)))
#     # ret /= float(height * width)
#     return ret
# #
#
def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]#
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm

#
def chara_image(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    Pi = hist/(1716*1438)
    I = np.arange(256).reshape(256, 1)
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
def get_image(image_name):
    # img = cv.imread(image_name)
    # img_shape = img.shape
    # print(img_shape)
    # img = cv.resize(img, (img_shape[1] / 2, img_shape[0] / 2), interpolation=cv.INTER_CUBIC)

    # img_gray = cv.cvtColor(image_name, cv.COLOR_BGR2GRAY)
    # cv.imshow('des', img_gray)
    # glcm_0 = getGlcm(image_name, 1, 0)
    glcm_1=getGlcm(image_name, 0,1)
    # glcm_2=getGlcm(image_name, 1,1)
    # glcm_3=getGlcm(image_name, -1,1)

    # asm0, con0, eng0, idm0 = feature_computer(glcm_0)
    asm1, con1, eng1, idm1 = feature_computer(glcm_1)
    # asm2, con2, eng2, idm2 = feature_computer(glcm_2)
    # asm3, con3, eng3, idm3 = feature_computer(glcm_3)
    return asm1, con1, eng1, idm1
#
#
#
chara_1 = []
chara_2 = []
# chara_3 = []


def batch_image(path_list):
    global chara_1
    global chara_2
    a = 0
    # global chara_3
    chara = []
    for i in range(2):
        for infer_path in path_list[i]:
            a = a+1
            print(a)
            if i == 0:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\50-60-train' + '/' + infer_path
                # sup = [0, 0, 1]
            if i == 1:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\50-60-test' + '/' + infer_path
                # sup = [0, 0, 1]
            # if i == 2:
            #     infer_path_1 = r'E:\pycharm\tensorflow\data\sort1/test_40' + '/' + infer_path
            #     sup = [0, 0, 1]
            # print(infer_path_1)
            blur_image = pre_image(infer_path_1)#预处理
            chara.append(chara_image(blur_image))#V值6个特征提取
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
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_01.txt", chara_1)#保存数据
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_01.txt", chara_2)
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt", chara_3)
print(chara_1[0])
print(chara_2[0])
# print(chara_3[0])

train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_01.txt")
train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_01.txt")
# train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt")
print(train_chara_1.shape)
print(train_chara_2.shape)
# print(train_chara_3.shape)
cv.waitKey(0)
cv.destroyAllWindows()

# infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\50-60-train' + '/' + path_list[0][0]
# t1 = cv.getTickCount()
# blur_image = pre_image(infer_path_1)
# # result = get_image(blur_image)
# cv.imshow('scr', blur_image)
# t2 = cv.getTickCount()
# time = (t2 - t1)/cv.getTickFrequency()
# # print(result)
# print(time)
# # src = cv.imread(infer_path_1)
# # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# # plt.figure(4)
# # plot_demo(gray)
# # plt.show()
# cv.waitKey(0)
# cv.destroyAllWindows()