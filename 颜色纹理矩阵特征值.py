#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2 as cv
import math
import numpy as np
import os
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



def equalHist_demo(image):#均衡化
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


def pre_image(infer_path):#图片预处理
    src = cv.imread(infer_path)#读取图片
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    # plot_demo(gray)
    src1 = src[:, :, 0]
    src2 = src[:, :, 1]
    src3 = src[:, :, 2]
    # gray1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    # gray2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    # gray3 = cv.cvtColor(src3, cv.COLOR_BGR2GRAY)
    equ1 = equalHist_demo(src1)
    equ2 = equalHist_demo(src2)
    equ3 = equalHist_demo(src3)#各层图像均衡化
    blur1 = blur_demo(equ1)
    blur2 = blur_demo(equ2)
    blur3 = blur_demo(equ3)#各层图像均值滤波
    return blur1, blur2, blur3



def getGlcm(input1, input2, input3, d_x, d_y):#提取颜色共生矩阵
    srcdata1 = input1.copy()
    srcdata2 = input2.copy()
    srcdata3 = input3.copy()
    ret1 = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    ret2 = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    ret3 = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    # ret = np.zeros([16, 16])

    (height, width) = input1.shape
    # (height2, width2) = input2.shape
    # (height3, width3) = input3.shape

    max_gray_level = 256

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    # if max_gray_level > gray_level:
    #     for j in range(height):
    #         for i in range(width):
    #             srcdata[j][i] = (srcdata[j][i] * gray_level / max_gray_level)

    srcdata1 = (srcdata1.astype(float) * gray_level / max_gray_level).astype(int)#像素大小缩放
    srcdata2 = (srcdata2.astype(float) * gray_level / max_gray_level).astype(int)
    srcdata3 = (srcdata3.astype(float) * gray_level / max_gray_level).astype(int)

    if d_x>=0:#计算颜色共生矩阵
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows1 = srcdata1[j][i]
                cols1 = srcdata2[j + d_y][i + d_x]
                ret1[rows1][cols1] += 1.0
                rows2 = srcdata2[j][i]
                cols2 = srcdata3[j + d_y][i + d_x]
                ret2[rows2][cols2] += 1.0
                rows3 = srcdata3[j][i]
                cols3 = srcdata1[j + d_y][i + d_x]
                ret3[rows3][cols3] += 1.0
    if d_x<0:
        for j in range(height - d_y):
            for i in np.arange(-d_x, width):
                rows1 = srcdata1[j][i]
                cols1 = srcdata2[j + d_y][i + d_x]
                ret1[rows1][cols1] += 1.0
                rows2 = srcdata2[j][i]
                cols2 = srcdata3[j + d_y][i + d_x]
                ret2[rows2][cols2] += 1.0
                rows3 = srcdata3[j][i]
                cols3 = srcdata1[j + d_y][i + d_x]
                ret3[rows3][cols3] += 1.0


    for i in range(gray_level):
        for j in range(gray_level):
            ret1[i][j] /= float((height - d_y) * (width - abs(d_x)))
            ret2[i][j] /= float((height - d_y) * (width - abs(d_x)))
            ret3[i][j] /= float((height - d_y) * (width - abs(d_x)))
    # ret /= float(height * width)
    return ret1, ret2, ret3
#
#
def feature_computer(p):
    # Con = 0.0
    Eng = 0.0
    # Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            # Con += (i - j) * (i - j) * p[i][j]
            # Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))#逆差值
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])#熵值
    # return Asm, Con, -Eng, Idm
    return -Eng/Idm#纹理复杂度系数

#
def get_image(blur1, blur2, blur3):#计算4个方向，每个方向3个共计12个纹理复杂度系数值
    # img = cv.imread(image_name)
    # img_shape = img.shape
    # print(img_shape)
    # img = cv.resize(img, (img_shape[1] / 2, img_shape[0] / 2), interpolation=cv.INTER_CUBIC)

    # img_gray = cv.cvtColor(image_name, cv.COLOR_BGR2GRAY)
    # cv.imshow('des', img_gray)
    glcm1_0,glcm2_0,glcm3_0=getGlcm(blur1, blur2, blur3, 1, 0)
    glcm1_1,glcm2_1,glcm3_1=getGlcm(blur1, blur2, blur3, 0,1)
    glcm1_2,glcm2_2,glcm3_2=getGlcm(blur1, blur2, blur3, 1,1)
    glcm1_3,glcm2_3,glcm3_3=getGlcm(blur1, blur2, blur3, -1,1)
    T11 = feature_computer(glcm1_0)
    T12 = feature_computer(glcm2_0)
    T13 = feature_computer(glcm3_0)
    # print(glcm1_1)

    T21 = feature_computer(glcm1_1)
    # print(glcm1_1)
    T22 = feature_computer(glcm2_1)
    T23 = feature_computer(glcm3_1)
    T31 = feature_computer(glcm1_2)
    T32 = feature_computer(glcm2_2)
    T33 = feature_computer(glcm3_2)
    T41 = feature_computer(glcm1_3)
    T42 = feature_computer(glcm2_3)
    T43 = feature_computer(glcm3_3)
    # asm10, con10, eng10, idm10 = feature_computer(glcm1_0)
    # asm20, con20, eng20, idm20 = feature_computer(glcm2_0)
    # asm30, con30, eng30, idm30 = feature_computer(glcm3_0)

    # asm11, con11, eng11, idm11 = feature_computer(glcm1_1)
    # asm21, con21, eng21, idm21 = feature_computer(glcm2_1)
    # asm31, con31, eng31, idm31 = feature_computer(glcm3_1)
    #
    # asm12, con12, eng12, idm12 = feature_computer(glcm1_2)
    # asm22, con22, eng22, idm22 = feature_computer(glcm2_2)
    # asm32, con32, eng32, idm32 = feature_computer(glcm3_2)
    #
    # asm13, con13, eng13, idm13 = feature_computer(glcm1_3)
    # asm23, con23, eng23, idm23 = feature_computer(glcm2_3)
    # asm33, con33, eng33, idm33 = feature_computer(glcm3_3)
    # return asm10, con10, eng10, idm10 ,asm20, con20, eng20, idm20, asm30, con30, eng30, idm30
    return T11, T12, T13, T21, T22, T23, T31, T32, T33, T41, T42, T43

#
#
#
chara_1 = []
chara_2 = []
# chara_3 = []


def batch_image(path_list):
    global chara_1
    global chara_2
    # global chara_3
    chara = []
    a=0
    for i in range(2):
        for infer_path in path_list[i]:
            a=a+1
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
            blur1, blur2, blur3 = pre_image(infer_path_1)#图片预处理
            chara.append(get_image(blur1, blur2, blur3))#提取12个颜色纹理特征
            # chara.append(sup)
        chara = np.hstack(chara).reshape(len(path_list[i]) , 12)
        print(chara.shape)
        if i == 0:
            chara_1 = chara.copy()
        if i == 1:
            chara_2 = chara.copy()

        # if i == 2:
        #     chara_3 = chara.copy()
        chara = []
# t1 = cv.getTickCount()

# infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\40-50-train' + '/' + path_list[0][0]
# blur1, blur2, blur3 = pre_image(infer_path_1)
# blur = np.zeros([1438, 1716, 3])
# print(blur.shape)
# blur[:,:,0] = blur1
# blur[:,:,1] = blur2
# blur[:,:,2] = blur3
# print(blur1.shape)
# print(blur2.shape)
# print(blur3.shape)
# print(blur.shape)
# # asm10, con10, eng10, idm10 ,asm20, con20, eng20, idm20, asm30, con30, eng30, idm30 = get_image(blur1, blur2, blur3)
#
# T11, T12, T13, T21, T22, T23, T31, T32, T33, T41, T42, T43 = get_image(blur1, blur2, blur3)
# print(T11, T12, T13, T21, T22, T23, T31, T32, T33, T41, T42, T43)
# t2 = cv.getTickCount()
# time = (t2 - t1)/cv.getTickFrequency()
# # print(result)
# print(time)
batch_image(path_list)#提取颜色纹理特征，得到特征值存储在chara_1和chara_2中
# # np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_1.txt", chara_1)
# # np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_2.txt", chara_2)
# # np.savetxt(r"E:\pycharm\tensorflow\data\sort1\train_chara_3.txt", chara_3)
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_5.txt", chara_1)#保存数据
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_5.txt", chara_2)
# # np.savetxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt", chara_3)
# print(chara_1[0])
# print(chara_2[0])
# # print(chara_3[0])
#
train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50-60_whole_5.txt")
train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50-60_whole_5.txt")
# # train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt")
print(train_chara_1.shape)
print(train_chara_2.shape)
print(train_chara_1[0])
print(train_chara_2[0])
# # print(train_chara_3.shape)
# cv.waitKey(0)
# cv.destroyAllWindows()

# infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\10-20-train' + '/' + path_list[0][0]
# t1 = cv.getTickCount()
# blur_image = pre_image(infer_path_1)
# result = get_image(blur_image)
# t2 = cv.getTickCount()
# time = (t2 - t1)/cv.getTickFrequency()
# print(result)
# print(time)
cv.waitKey(0)
cv.destroyAllWindows()