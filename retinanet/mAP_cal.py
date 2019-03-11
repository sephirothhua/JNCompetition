# -*- coding: UTF-8 -*-
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
from PIL import Image
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer,JinNanDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage
import json

#assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class my_eval(COCOeval):
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
            #     g['_ignore'] = 1
            # else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

def get_eval(cocoGt,resFile,evaltype='bbox'):
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = my_eval(cocoGt, cocoDt, evaltype)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def get_cocoGT(annFile):
    cocoGt = COCO(annFile)
    return cocoGt
def read_model(model_path):
    retinanet = torch.load(model_path)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()
    return retinanet

class Norm(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image):

        return (image.astype(np.float32)-self.mean)/self.std

class Resize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=608, max_side=1024):

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)


        return torch.from_numpy(new_image)

transform1 = transforms.Compose([Norm(), Resize()])


def transform_image(image):
    image = np.array(image) / 255.0
    image = transform1(image)
    image = image[np.newaxis,:]
    image = np.transpose(image,(0,3,1,2))
    return image



def detect_images(model,cocoGt,data_path,limit=0.5):
    result_list = []
    with torch.no_grad():
        for id in cocoGt.getImgIds():
            Img = cocoGt.loadImgs(id)[0]
            image = Image.open(os.path.join(data_path, Img['file_name']))

            scores, classification, transformed_anchors = model(transform_image(image).cuda().float())
            idxs = np.where(scores.cpu().numpy() > limit)
            for j in idxs[0]:
                bbox = transformed_anchors[j, :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cls = int(classification[j])
######################################Show the image#########################################################
                img = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_RGB2BGR)

                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                cv2.imshow('img', img)
                cv2.waitKey(0)
#############################################################################################################
                result_list.append(
                    {"image_id": id, "category_id": cls, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                     "score": float(scores[j])})
                print("Processing {}".format(id))
    return result_list

def save_json(result_list,json_name):
    # 存储json文件
    json_str = json.dumps(result_list, cls=MyEncoder)
    with open('./eval/{}'.format(json_name), 'w') as json_file:
        json_file.write(json_str)


annFile = '/data/Project/Tianchi/JNCompetition/yolo3/data/train_0305/train_no_poly.json'
data_path = '/data/Project/Tianchi/JNCompetition/yolo3/data/train_0305/restricted'
model_dir = 'model_final.pt'
resFile='./eval/result.json'
json_name = datetime.datetime.now().strftime('%m_%d_%H_%M_%S') + '.json'


if __name__ == '__main__':
    cocoGt = get_cocoGT(annFile)
    retina = read_model(model_dir)
    result = detect_images(retina,cocoGt,data_path)
    save_json(result,json_name)
    get_eval(cocoGt,'./eval/{}'.format(json_name))
    #print(result)
