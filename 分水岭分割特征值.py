import cv2 as cv
import numpy as np
import os
from collections import Counter
path_20_cut = r'E:\pycharm\tensorflow\data\sort2\10-20-train'
# path_20_40_cut = r'E:\pycharm\tensorflow\data\sort2\test_20_40'
path_40_cut = r'E:\pycharm\tensorflow\data\sort2\10-20-test'
path_list_30 = os.listdir(path_20_cut)
# path_list_40 = os.listdir(path_20_40_cut)
path_list_50 = os.listdir(path_40_cut)
# path_list = [path_list_30, path_list_40, path_list_50]
path_list = [path_list_30, path_list_50]

import math
#计算大小，扩充像素点
#制作一个7*7模板，看是否有检测到其他数值+
base_sign=[]
base_sign_1=[]
data = []#泡沫亮点个数/大泡沫亮点数/大泡沫平均大小/
def expand(markers):#根据标记逐个放大大泡沫
    w = markers.shape[0]
    h = markers.shape[1]
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)  # 取得大泡沫标记
    # print(a)
    sum = 0
    src1 = np.zeros([w, h])
    for i in range(1, a[-1]+1):
        index = np.argwhere(markers == i)  # 得到该大泡沫的所有像素位置索引
        num = int(index.shape[0]**0.33)*8#根据像素个数用函数放大
        # print(num)
        sum = sum +num#放大后大泡沫的像素个数
        # print(num)
        index = np.mean(index, axis=0)#得到中心点
        cv.circle(src1, center=(int(index[1]), int(index[0])), radius=num, color=255, thickness=-1)#在各个大泡沫中心点画圆放大，半径为num
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # # mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    # sure_bg = cv.dilate(surface, kernel, iterations=20)
    # cv.imshow("markers1", src1)
    src1 = np.uint8(src1)
    # print(sum)
    # print(a[-1])
    sum = float(sum)/float(a[-1])#大泡沫平均大小
    # ret, markers = cv.connectedComponents(src1)
    # a = np.unique(src1, return_index=False, return_inverse=False, return_counts=False)  # 取得大泡沫标记
    # print(a)
    return src1,sum

def etch(img, step, step1):#选择疏松区域膨胀
    # h=img.shape[0]
    # w=img.shape[1]
    img1=img.copy()
    # a = np.unique(img, return_index=False, return_inverse=False, return_counts=False)
    # print(a)
    # a_last = 6
    global base_sign#紧密区域标记
    for i in range (step1,499-step1,step):#遍历像素
        for j in range (step1,499-step1,step):
            base=img[i,j]
            a=0
            for k in range (i-step1,i+step1+1):#x方向判断是否紧密
                for l in range (j,j+1):
                    # if k<0|k>=h-1|l<0|l>=w-1:
                    #     continue
                    # print(k)
                    # print(l)
                    if img[k,l]!=0 and img[k,l]!=base:
                        a = a+1
            for k1 in range(i , i + 1):#y方向判断是否紧密
                for l1 in range(j-step1, j + step1+1):
                    # if k<0|k>=h-1|l<0|l>=w-1:
                    #     continue
                    # print(k)
                    # print(l)
                    # if img[k,l]!=0 and img[k,l]!=base:
                    if img[k1, l1] != 0 and img[k1, l1] != base:
                        a = a + 1
            for k2 in range(-step1 , step1+1):#45/135方向判断是否紧密
                k2_1 = k2 + i
                l2_1 = k2 + j
                k2_2 = -k2 + i
                l2_2 = -k2 + j
                if (img[k2_1, l2_1] != 0 and img[k2_1, l2_1] != base) or (img[k2_2, l2_2] != 0 and img[k2_2, l2_2] != base) :
                    a = a + 1
                    # if k<0|k>=h-1|l<0|l>=w-1:
                    #     continue
                    # print(k)
                    # print(l)
                    # if img[k,l]!=0 and img[k,l]!=base:
                    # if img[k, l] != 0 and img[k, l] != base:
                    #     a = a + 1
            # for k in range(i - 8, i + 9):
            #     for l in range(j, j + 1):
            #         # if k<0|k>=h-1|l<0|l>=w-1:
            #         #     continue
            #         # print(k)
            #         # print(l)
            #         # if img[k,l]!=0 and img[k,l]!=base:
            #         a = a + 1
                        # a_now = min(abs(k - i), abs(l - j), a_last)
                        # a_last = a_now
            # print(base)
            # print(a_last)
            # print(a==0&base!=0)
            # print(a)
            # print(a == 0 and base != 0)
            # if a_last==4:
            #     for k1 in range(i - 1, i + 2):
            #         for l1 in range(j - 1, j + 2):
            #             img1[k1,l1]=base
            if a!=0 and base not in base_sign:
                base_sign.append(base)
            if a==0 and base!=0 and base not in base_sign:#膨胀疏松像素
                for k2 in range(i - 10, i + 11):
                    for l2 in range(j - 10, j + 11):
                        img1[k2,l2]=base

            a = 0
            # print(base)

    return img1


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)#高斯滤波
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)#灰度化
    # X Gradient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    #edge
    #edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)#canny边缘提取
    # cv.imshow("Canny Edge", edge_output)

    dst = cv.bitwise_and(image, image, mask=edge_output)
    # cv.imshow("Color Edge", dst)
    return edge_output
def chance (markers):
    markers1 = etch(markers,  5, 25)#膨胀，跨5步，模型长度为25
    markers_last = etch(markers1,  -5, 25)
    return markers_last
def watershed_demo(src1):#分水岭特征提取
    w = src1.shape[0]
    h = src1.shape[1]
    src2 = src1
    blur = blur_demo(src2)#边缘保留滤波
    edge = edge_demo(blur)#canny算法
    ret, markers = cv.connectedComponents(edge)#计算canny算法的连通区域个数
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)

    data.append(max(a)/1000)#特征1：canny连通区域个数
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)#灰度化
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#二值化
    ret, markers = cv.connectedComponents(binary)
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)#初始化后连通区域个数
    # print(a)
    data.append(max(a)/1000)#特征2：边缘保留滤波二值化后的连通区域个数
    src1 = np.zeros([w, h])
    src1[markers!=0] = 255
    # cv.imshow("markers1", src1)                    #初始化后的不为0标记

    dist = cv.distanceTransform(binary, cv.DIST_L1, 3)#距离变换
    ret, surface = cv.threshold(dist, 5, 255, cv.THRESH_BINARY)#定义大泡沫阈值为5
    # cv.imshow("markers3", surface)
    surface = np.uint8(surface)
    ret, markers = cv.connectedComponents(surface)
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)#取得大泡沫标记
    # print(a)
    data.append(max(a))#特征3：大泡沫个数
    # cv.imshow("markers4", surface)
    src1,sum = expand(markers)#放大大泡沫
    data.append(sum)#特征4：大泡沫平均半径大小
    binary = binary|src1#大泡沫放大加入图像
    binary = np.uint8(binary)
    ret, markers = cv.connectedComponents(binary)
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)  # 取得大泡沫标记
    # print(a)
    # cv.imshow("markers5", binary)

    # print(len(base_sign))
    # markers_last = chance(markers_last)
    # print(len(base_sign))
    # # markers_last = chance(markers_last)
    # # print(len(base_sign))
    # # markers_last = chance(markers_last)
    # # print(len(base_sign))
    # # markers_last = chance(markers_last)
    # # print(len(base_sign))
    # # markers_last = chance(markers_last)
    # # markers_last = chance(markers_last)
    #
    # # markers4 = etch(markers3, 5, 3)
    # print(np.sum(markers > 0))
    # print(np.sum(markers_last > 0))
    # # print(np.sum(markers2 > 0))
    # # print(np.sum(markers3 > 0))
    # src1 = np.zeros([500, 500])
    # src1[markers_last != 0] = 255
    # cv.imshow("markers4", src1)
    markers = np.uint8(markers)
    ret, binary = cv.threshold(markers, 0, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("binary-image1", binary)  # 求反为了距离变换

    #
    #
    #
    #
    # src1 = np.uint8(src1)
    # ret, markers = cv.connectedComponents(src1)  # 取得标记154个
    # a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)
    # print(a)
    # # # morphology operation
    #
    # # sure_bg = cv.dilate(binary, kernel, iterations=8)
    # # cv.imshow("mb", mb)
    # # cv.imshow("sure_bg", sure_bg)
    #
    # # distance transform
    dist = cv.distanceTransform(binary, cv.DIST_L1, 3)#距离变换
    B = np.unique(dist, return_index=False, return_inverse=False, return_counts=False)
    # print(B)

    # #
    ret, surface = cv.threshold(dist, 2, 255, cv.THRESH_BINARY_INV)#所有图像扩大两个像素
    # print(ret)
    surface_fg = np.uint8(surface)
    # cv.imshow("surface-bin", surface_fg)
    # unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)#得到分水岭标记
    a = np.unique(markers, return_index=False, return_inverse=False, return_counts=False)
    data.append(max(a)/1000)#特征5：分水岭标记个数
    markers = chance(markers)#膨胀疏松区域
    src1 = np.zeros([w, h])
    src1[np.uint8(markers) !=0] = 255
    # cv.imshow("markers1", src1)
    markers = cv.watershed(blur, markers=markers)#膨胀后图像的连通区域作为分水岭的标记
    blur[markers==-1] = [0, 0, 255]
    # cv.imshow("result", blur)
    # src1 = np.zeros([500, 500])
    # cv.imshow("markers22", src2)
    src2[markers==-1] = [0, 0, 255]
    # cv.imshow("markers2", src2)
    markers1 = np.hstack(markers)
    markers1 = Counter(np.array(markers1))#计算分水岭后各个标记个数
    a = list(markers1.values())
    # print(markers1[-1])
    data.append(markers1[-1]/1000)#特征6：分水岭后分割线长度
    # a = min(markers1.values())
    # print(sorted(a))
    arr_mean = np.mean(a)
    data.append(arr_mean / 1000)#特征7：各个标记区域像素个数均值
    # print(arr_mean)
    #求方差
    arr_var = np.var(a)
    data.append(arr_var / 1000000)#特征8：各个标记区域像素个数方差
    #求标准差
    arr_std = np.std(a,ddof=1)
    data.append(arr_std / 1000)#特征9：各个标记区域像素个数标准差
    # print(arr_var)
    # print(arr_std)
    return data

def equalHist_demo(image):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(image)
    # cv.imshow("equalHist_demo", dst)
    # plot_demo(dst)
    return dst




def blur_demo(image):#边缘保留滤波
    # dst = cv.pyrMeanShiftFiltering(image, 10, 100)#154
    # dst = cv.pyrMeanShiftFiltering(image, 5, 50)#282
    dst = cv.pyrMeanShiftFiltering(image, 5, 15)  # 282
    return dst


def pre_image(src1):
    src = src1
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("src", src)
    blur = blur_demo(src)
    cv.imshow("blur image", blur)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    # plot_demo(gray)
    # equ = equalHist_demo(gray)
    # cv.imshow("equ", equ)

    # cv.imshow("blur", equ)
    return gray

chara_1 = []
chara_2 = []
# chara_3 = []


def batch_image(path_list):
    global chara_1
    global chara_2
    global data
    a = 0
    # global chara_3
    chara = []
    for i in range(2):
        for infer_path in path_list[i]:
            a = a+1
            print(a)
            if i == 0:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\10-20-train' + '/' + infer_path
                # sup = [0, 0, 1]
            if i == 1:
                infer_path_1 = r'E:\pycharm\tensorflow\data\sort2\10-20-test' + '/' + infer_path
                # sup = [0, 0, 1]
            # if i == 2:
            #     infer_path_1 = r'E:\pycharm\tensorflow\data\sort1/test_40' + '/' + infer_path
            #     sup = [0, 0, 1]
            # print(infer_path_1)
            src = cv.imread(infer_path_1)#读取图片
            src = src[219:1219, 358:1358]#选择1000*1000个像素点
            chara.append(watershed_demo(src))#分水岭分割后的特征提取9个
            print(data)
            data = []
            # chara.append(sup)
        chara = np.hstack(chara).reshape(len(path_list[i]) , 9)
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
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_10_20_whole_size.txt", chara_1)#保存数据
np.savetxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_10_20_whole_size.txt", chara_2)
# np.savetxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt", chara_3)
print(chara_1[0])
print(chara_2[0])
# print(chara_3[0])

train_chara_1 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\train_chara_10_20_whole_size.txt")
train_chara_2 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort2\test_chara_10_20_whole_size.txt")
# train_chara_3 = np.loadtxt(r"E:\pycharm\tensorflow\data\sort1\test_chara_3_whole.txt")
print(train_chara_1.shape)
print(train_chara_2.shape)
# print(train_chara_3.shape)
cv.waitKey(0)
cv.destroyAllWindows()
