# -*- coding=utf-8 -*-
import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from kmeans import kmeans, avg_iou
from PIL import Image

# 根文件夹
ROOT_PATH = './data/'
label_file = 'train_r1.txt'
# 聚类的数目
CLUSTERS = 9
# 模型中图像的输入尺寸，默认是一样的
SIZE = 640

# 加载YOLO格式的标注数据
def load_dataset(path):
    dataset = []
    # for label in label_file:
    with open(path, 'r') as f:
        txt_content = f.readlines()
        print('label count: {}'.format(len(txt_content)))

    for line in txt_content:
        line_split = line.split(' ')
        dir = line_split[0]
        image = Image.open(dir)
        height = image.height
        width = image.width
        for i in line_split[1:]:
            x1,y1,x2,y2,cls = i.split(',')
            roi_with = (float(x2) - float(x1))/width
            roi_height = (float(y2) - float(y1))/height
            if roi_with == 0 or roi_height == 0:
                continue
            dataset.append([roi_with, roi_height])
        # print([roi_with, roi_height])

    return np.array(dataset)

def sum_pics_heigth_width(path):
    dataset = []
    # for label in label_file:
    with open(path, 'r') as f:
        txt_content = f.readlines()
        print('label count: {}'.format(len(txt_content)))

    for line in txt_content:
        line_split = line.split(' ')
        dir = line_split[0]
        image = Image.open(dir)
        height = image.height
        width = image.width
        dataset.append([width, height])
        # print([roi_with, roi_height])

    return np.array(dataset)

data = load_dataset(os.path.join(ROOT_PATH,label_file))
out = kmeans(data, k=CLUSTERS)

print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}-{}".format(out[:, 0] * SIZE, out[:, 1] * SIZE))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))


# Get The Mean Of All Picture Width And Height
# data = sum_pics_heigth_width(os.path.join(ROOT_PATH,label_file))
# mean = np.mean(data,axis=0)
# print(mean)
