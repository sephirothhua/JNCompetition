from test_frcnn_kitti import *
import json
import os
import numpy as np
from PIL import Image
import cv2


# annotation_path = './data/train_r1.txt'

class_names = ['fefire','blfire','knife','bat','shear']

cls_dict = {
    0:3,
    1:5,
    2:4,
    3:1,
    4:2
}

data_path = "/home/zhang/桌面/jinnan/jinnan2_round1_test_a_20190306/"




class_dict = {"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0}
result_dict = {'results':[]}
result = []

def get_result(pic,model_rpn, model_classifier, cfg,SavePic=False):

    image = cv2.imread(os.path.join(data_path,pic))
    #print(image.shape)
    cls, box_real, prop = predict_one_image(model_rpn, model_classifier, cfg, image)
    if SavePic:
        img = cv2.imread(os.path.join(data_path,pic))
    r_dict = {}
    r_list = []
    r_dict['filename']=pic
    for i in range(len(cls)):
        x1,y1,x2,y2 = box_real[i]
        clas=cls[i]
        print('class:',clas)
        score = prop[i]
        r_list.append({'xmin':x1,'xmax':x2,'ymin':y1,'ymax':y2,'label':cls_dict[clas],'confidence':score})
        if SavePic:
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            #print("%s %.3f"%(class_names[clas],prop[i]))
            cv2.putText(img, "%s %.3f"%(class_names[clas],prop[i]), (int(x1), int(y1) - 1), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0),2)
            cv2.imwrite(os.path.join("./results_images",pic),img)
    r_dict['rects']=r_list
    return r_dict



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


if __name__ == '__main__':
    model_rpn, model_classifier, cfg = load_model()
    for a, b, c in os.walk(data_path):
        for pic in c:
            r = get_result(pic,model_rpn, model_classifier, cfg ,SavePic=True)
            #print(r)
            result.append(r)
    result_dict['results'] = result
    json_str = json.dumps(result_dict, cls=MyEncoder)
    with open('./result/true_last.json', 'w') as json_file:
        json_file.write(json_str)
        print('finish write!')
    #file = open("./result/last.json", 'rb')
    #a_dict = json.load(file)
    #print(a_dict)