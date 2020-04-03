import pandas as pd
import os
import numpy as np
import cv2 as cv
import shutil

import random
from shutil import copyfile
import math
import datetime
#读取表格数据，根据usecols的A,B,C列划分
data = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="C", dtype={'ID': str, 'InStore': str, 'Date': str})#标题为第0行
# data = pd.read_excel('E:\pycharm\excel/12月品位数据.xlsx', usecols="C")#标题为第0行
data = np.array(data)
data_A = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="A")#标题为第0行
data_A = np.array(data_A)
data_B = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="B")#标题为第0行
data_B = np.array(data_B)
print(data.shape)



def get_value(index):
    image_value = []
    path = r'E:\pycharm\tensorflow\data\sort2'
    path_list = os.listdir(path)
    infer_path = path_list[index]#选择文件夹
    infer_path_1 = r'E:\pycharm\tensorflow\data\sort2' + '/' + infer_path
    path_list_1 = os.listdir(infer_path_1)
    # print(path_list_1[0])
    for index_1 in range(len(path_list_1)):#遍历文件夹中每一张图片
        date = path_list_1[index_1][6]+path_list_1[index_1][8]+path_list_1[index_1][9]#根据文件名得到日期信息
        # print(date)
        if path_list_1[index_1][11]=='0':
            hour = path_list_1[index_1][12]#时间信息
        else:
            hour = path_list_1[index_1][11]+path_list_1[index_1][12]#时间信息
        # print(hour)
        # print(hour)
        # print(data_A[0][0][5])
        for i in range(len(data_A)):
            if pd.isna(data_A[i]) == 0:#非空
                if data_A[i][0][5] == '7':
                    pre_day = data_A[i][0][5] + data_A[i][0][7] + data_A[i][0][8]
                else:
                    pre_day = data_A[i][0][5] + '0' + data_A[i][0][7]
                if date == pre_day:
                    index_day_1 = i#得到和图片日期信息相等的excel的日期信息索引
        # print(data_B[index_day_1][0].hour)
        # print(hour)
        # print(int(data_B[index_day_1][0].hour)!=int(hour))
        while int(data_B[index_day_1][0].hour)!=int(hour):
            # print(index_day_1)
            index_day_1 = index_day_1+1#得到和图片时间信息相等的excel的时间信息索引
        # print(index_day_1)
        print(data[index_day_1])
        image_value.append(data[index_day_1])#得到该图片品位值
    return image_value


image_value_test = get_value(17)
image_value_train = get_value(18)

print(len(image_value_train))
print(len(image_value_test))
image_value_train = np.array(image_value_train).reshape(len(image_value_train), 1)
image_value_test = np.array(image_value_test).reshape(len(image_value_test), 1)
print(image_value_train.shape)
print(image_value_test.shape)
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_label.txt", image_value_train)
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\teat_chara_50_60_whole_label.txt", image_value_test)#存起来
# # train_chara_1_spit = np.hsplit(train_chara_1, np.array([6]))
# # train_chara_2_spit = np.hsplit(train_chara_2, np.array([6]))
# # print(train_chara_1_spit[0].shape)
# # print(train_chara_2_spit[0].shape)
# # print(train_chara_1_spit[1].shape)
# # print(train_chara_2_spit[1].shape)
# train_chara_1_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_50_60_whole_45.txt")
# train_chara_2_4 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_50_60_whole_45.txt")
# image_path = r'E:/pycharm/tensorflow/data/sort2' + '/' + infer_path + '/'+ path_list_1[0]
# print(image_path)



# img = cv.imread(image_path)
# cv.imshow('scr', img)
# cv.waitKey()
# cv.destroyAllWindows()



