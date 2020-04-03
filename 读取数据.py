import pandas as pd
import os
import numpy as np
import shutil
import random
from shutil import copyfile
import math
import datetime
#根据excel的品位信息分类，得到每一类的时间信息。根据得到的时间信息将图片数据放在指定文件夹
#读取表格数据，根据usecols的A,B,C列划分
data = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="C", dtype={'ID': str, 'InStore': str, 'Date': str})#标题为第0行
# data = pd.read_excel('E:\pycharm\excel/12月品位数据.xlsx', usecols="C")#标题为第0行
data = np.array(data)
print(data.shape)
data_1 = []
data_2 = []
data_3 = []#存放各个分类的索引
#将表格数据划分为3组
for i in range(data.shape[0]):
    if data[i][0] <= 20:
        data_1.append(i)
    if 20 < data[i][0] <40:
        data_2.append(i)
    if 40 <= data[i][0]:
        data_3.append(i)
print(len(data_1))
print(len(data_2))
print(len(data_3))
data_1_A = data_1.copy()
data_2_A = data_2.copy()
data_3_A = data_3.copy()
data_A = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="A")#标题为第0行，日期信息
data_A = np.array(data_A)
data_B = pd.read_excel('E:\pycharm\excel/7月.xlsx', usecols="B")#标题为第1行，时间信息
data_B = np.array(data_B)
# print(data_B)

#得到每组的图像数据
def get_img(data_1, data_1_A):
    # print(len(data_1))
    # print(len(data_1_A))
    for i in range(len(data_1_A)):#得到日期信息
        while pd.isna(data_A[data_1_A[i]])==1:
            data_1_A[i] = data_1_A[i]-1
    # print(len(data_1_A))
    path = r'E:\pycharm\tensorflow\data\7月'
    path_list = os.listdir(path)
    path_list = np.array(path_list)
    index_day_1 = []#得到原未分类图像存储日期索引数据
    # print(data_A[data_1_A])
    # print(data_A[data_1_A].shape)
    for i in range(len(data_1_A)):#根据得到的日期信息找到对应的原图像文件夹
        if data_A[data_1_A][i][0][5]=='7':
            pre_day = data_A[data_1_A][i][0][5] + data_A[data_1_A][i][0][7] + data_A[data_1_A][i][0][8]
        else:
            # print(8)
            pre_day = data_A[data_1_A][i][0][5] + '0' + data_A[data_1_A][i][0][7]
            # print(pre_day)
        # print(pre_day)
        index_day_1.append(np.where(path_list == pre_day))
    # print(index_day_1)
    # print(data_A[data_1_A])
    # print(data_B[data_1])
    data_1_img = []#存储符合小时数的图像数据
    for i in range(len(data_1)):
        pre_hour = data_B[data_1][i][0].hour#excel中的小时信息
        # pre_min = data_B[data_1][i][0].minute
        # print(pre_hour)
        # print(pre_min)
        infer_path = r'E:\pycharm\tensorflow\data\7月' + '/' + path_list[index_day_1[i]][0]#对应日期的文件夹
        path_list_1 = os.listdir(infer_path)
        path_list_1 = np.array(path_list_1)#对应日期的所有图片文件名
        # print(path_list_1.shape[0])
        a = len(data_1_img)#记录当前数量
        for j in range(path_list_1.shape[0]):#遍历该文件夹所有图片
            real_hour = path_list_1[j][11]+path_list_1[j][12]#图片文件名中的时间信息
            # real_min = path_list_1[][11]+path_list_1[0][12]
            if int(pre_hour) == int(real_hour):
                infer_path = r'E:\pycharm\tensorflow\data\7月' + '/' + path_list[index_day_1[i]][0] + '/' +path_list_1[j]
                data_1_img.append(infer_path)
        if len(data_1_img)-a==0:
            print("漏掉的是%d", i)#输出在excel中有日期而对应日期在真实数据中没有的索引
    # print(path_list)
    a = 0
    for i in range(path_list.shape[0]):#计算所有文件图像总数
        infer_path = r'E:\pycharm\tensorflow\data\7月' + '/' + path_list[i]
        path_list_1 = os.listdir(infer_path)
        # print(len(path_list_1))
        a = a + len(path_list_1)
    print(a)
    return data_1_img

#将图像文件复制到指定路径
def assign_data(data_img_1, path_20):
    for i in range(len(data_img_1)):
        shutil.copy(data_img_1[i], path_20)


path_20 = r'E:\pycharm\tensorflow\data\sort1\train_20'#品位值为20以下的训练集图片存放位置
path_20_40 = r'E:\pycharm\tensorflow\data\sort1\train_20_40'
path_40 = r'E:\pycharm\tensorflow\data\sort1\train_40'
path_20_test = r'E:\pycharm\tensorflow\data\sort1\test_20'#品位值为20以下的测试集图片存放位置
path_20_40_test = r'E:\pycharm\tensorflow\data\sort1\test_20_40'
path_40_test = r'E:\pycharm\tensorflow\data\sort1\test_40'

data_img_1 = get_img(data_1, data_1_A)
print(len(data_img_1))
data_img_2 = get_img(data_2, data_2_A)
print(len(data_img_2))
# # print(data_2[40])
# # # print(data_2[41])
# # # print(data_2[42])
# # # print(data_2[43])
data_img_3 = get_img(data_3, data_3_A)
print(len(data_img_3))
# print(len(data_img_1)+len(data_img_2)+len(data_img_3))
# assign_data(data_img_1, path_20)
# assign_data(data_img_2, path_20_40)
# assign_data(data_img_3, path_40)

#将符合要求的数据划分为train和test数据集
def get_test(data_img_1):
    l1 = int(len(data_img_1)*0.3)
    test_1 = []
    for i in range(l1):
        a = random.sample(range(0, len(data_img_1)), 1)
        test_1.append(data_img_1.pop(a[0]))
    return data_img_1, test_1


data_img_1, test_1 = get_test(data_img_1)
data_img_2, test_2 = get_test(data_img_2)
data_img_3, test_3 = get_test(data_img_3)
print(len(data_img_1))
print(len(data_img_2))
print(len(data_img_3))
print(len(test_1))
print(len(test_2))
print(len(test_3))
assign_data(data_img_1, path_20)
assign_data(data_img_2, path_20_40)
assign_data(data_img_3, path_40)
assign_data(test_1, path_20_test)
assign_data(test_2, path_20_40_test)
assign_data(test_3, path_40_test)

# data_img_1 = np.array(data_img_1)
# data_img_1_test = data_img_1[a]
# shutil.move(data_img_1_test, path_20_test)
# print(data_B[data_2])
# for i in range(len(data_2)):
#     while pd.isna(data_A[data_2[i]])==1:
#         data_2[i] = data_2[i]-1
# print(data_A[data_2])
#
# print(data_B[data_3])
# for i in range(len(data_3)):
#     while pd.isna(data_A[data_3[i]])==1:
#         data_3[i] = data_3[i]-1
# print(data_A[data_3])





