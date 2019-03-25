import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


def iou(bb_test, bb_gt):
    '''
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    Parameters:
        bb_test: [x1,y1,x2,y2,...]
        bb_ground: [x1,y1,x2,y2,...]
    Returns:
        score: float, takes values between 0 and 1.
        score = Area(bb_test intersects bb_gt)/Area(bb_test unions bb_gt)
    '''
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    area = w * h
    score = area / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                    + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - area)
    return score


iou_thresh = 0.6
def assign(predict_boxes, real_boxes):
    iou_metric = []
    for box in predict_boxes:
        temp_iou = []
        for box2 in real_boxes:
            temp_iou.append(iou(box, box2))
        iou_metric.append(temp_iou)
    iou_metric = np.array(iou_metric)
    result = linear_assignment(-iou_metric)
    output = []
    output_iou = []
    for idx in range(len(result)):
        if iou_metric[result[idx][0],result[idx][1]] > iou_thresh:
            output.append(result[idx])
            output_iou.append(iou_metric[result[idx][0],result[idx][1]])
    return output, output_iou


#       predict
#     yes    no
# yes  TP    FN    real
# no   FP    TN

# acc = (TP + TN)/(TP+FN+FP+TN)
# recall = TP/(TP + FN)
#
# 调节score阈值，算出召回率从0到1时的准确率，得到一条曲线
# 计算曲线的下面积 则为AP


def get_auc(xy_arr):
    # 计算曲线下面积即AUC
    auc = 0.
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    # 画出auc图
    # plt.ylabel("False Positive Rate")
    # plt.plot(x, y)
    # plt.show()
    # print(xy_arr)
    return auc

def caculate_AP(predict_boxes, real_boxes):
    recall_arr = []
    acc_arr = []
    xy_arr = []
    score_arr = list(map(lambda input:float(input)*0.01, range(0, 101)))
    for score in score_arr:
        temp_predict_boxes = []
        for box in predict_boxes:
            if box[4]>score:
                temp_predict_boxes.append(box)
        result,_ = assign(temp_predict_boxes, real_boxes)
        TP = len(result)
        FN = len(real_boxes) - TP
        FP = len(temp_predict_boxes) - TP
        recall = TP/(TP+FN)
        acc = TP/(TP+FN+FP)
        recall_arr.append(recall)
        acc_arr.append(acc)
        xy_arr.append([recall,acc])
    return get_auc(xy_arr)


def get_mAP(all_predict_boxes, all_real_boxes):
    ap_arr = []
    for idx in range(len(all_predict_boxes)):
        ap_arr.append(caculate_AP(all_predict_boxes[idx], all_real_boxes[idx]))
    return np.mean(ap_arr)


if __name__ == '__main__':
    print(get_mAP(np.random.uniform(0, 1, (100, 10, 5)), np.random.uniform(0, 1, (100, 10, 4))))
