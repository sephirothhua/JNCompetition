import os
from test_frcnn_kitti import *
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import datetime
import cv2


cls_dict = {
    0:3,
    1:5,
    2:4,
    3:1,
    4:2
}

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

def get_cocoGT(annFile):
    cocoGt = COCO(annFile)
    return cocoGt

def get_result2json(cocoGt,data_path,model_rpn, model_classifier, cfg,json_name):
    result_list = []
    for i,id in enumerate(cocoGt.getImgIds()):
        Img = cocoGt.loadImgs(id)[0]
        image = cv2.imread(os.path.join(data_path, Img['file_name']))
        print("\r {}/{}".format(i,len(cocoGt.getImgIds())))
        # box_result, score_result = yolo.detect(image)
        classes, box_real, prop = predict_one_image(model_rpn, model_classifier, cfg, image)
        for i in range(len(classes)):
            x1, y1, x2, y2 = box_real[i]
            score = prop[i]
            cls = classes[i]
            result_list.append(
                {"image_id": id, "category_id": cls_dict[cls], "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                 "score": score})
    # 存储json文件
    json_str = json.dumps(result_list, cls=MyEncoder)
    with open('./eval/{}'.format(json_name), 'w') as json_file:
        json_file.write(json_str)
    return './eval/{}'.format(json_name)


def get_eval(cocoGt,resFile,evaltype='bbox'):
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = my_eval(cocoGt, cocoDt, evaltype)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


annFile = '/home/zhang/桌面/jinnan/jinnan2_round1_train_20190305/train_no_poly.json'
data_path = '/home/zhang/桌面/jinnan/jinnan2_round1_train_20190305/restricted'
json_name = datetime.datetime.now().strftime('%m_%d_%H_%M_%S') + '.json'

if __name__ == '__main__':
    model_rpn, model_classifier, cfg = load_model()
    cocoGt = get_cocoGT(annFile)
    res_file = get_result2json(cocoGt,data_path,model_rpn, model_classifier, cfg,json_name)
    get_eval(cocoGt,res_file)