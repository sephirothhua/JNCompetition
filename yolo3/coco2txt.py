import json
from pycocotools.coco import COCO
import os
import re
def load_coco(dataset_dir, image_dir, class_ids=None,):
    coco = COCO("{}".format(dataset_dir))
    coco_list = []
    # Load all classes or a subset?
    if not class_ids:
        # All classes
        class_ids = sorted(coco.getCatIds())

    # All images or a subset?
    if class_ids:
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(coco.getImgIds(catIds=[id])))
        # Remove duplicates
        image_ids = list(set(image_ids))
    else:
        # All images
        image_ids = list(coco.imgs.keys())

    # # Add classes
    # for i in class_ids:
    #     add_class("coco", i, coco.loadCats(i)[0]["name"])

    # Add images
    for i in image_ids:
        coco_list.append(
            {
            'image_id':i,
            'path':os.path.join(image_dir, coco.imgs[i]['file_name']),
            'width':coco.imgs[i]["width"],
            'height':coco.imgs[i]["height"],
            'annotations':coco.loadAnns(coco.getAnnIds(
                imgIds=[i], catIds=class_ids, iscrowd=None))})
    return coco_list

def coco2txt(CocoList,TxtDir):
    file = open(TxtDir, 'r+')
    for item in CocoList:
        img_dir = item['path']
        messages = item['annotations']
        mes = ""
        for message in messages:
            x1,y1,w,h = message['bbox']
            class_id = message['category_id']
            mes += " {},{},{},{},{}".format(int(x1),int(y1),int(x1+w),int(y1+h),int(class_id-1))
        file.write('{}{}\n'.format(img_dir,mes))
    file.close()


# JSON_DIR = "./data/json/train_r1.json"
# PIC_DIR = "./data/train_r1"
# TxtDir = "./data/train_r1.txt"
# coco_all = load_coco(JSON_DIR,PIC_DIR)
# coco2txt(coco_all,TxtDir)


data_dir = "./data"
TxtDir = "./data/train_r1.txt"
coco_list = []
for root,dirs,files in os.walk(data_dir):
    for dir in dirs:
        if(dir.split('_')[0]=="train"):
            json_file = os.path.join(root,dir,"train_no_poly.json")
            pic_dir = os.path.join(root,dir,"restricted")
            coco1 = load_coco(json_file,pic_dir)
            coco_list.extend(coco1)

coco2txt(coco_list,TxtDir)
