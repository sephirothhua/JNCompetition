# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------

"""normal obj and restricted obj dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import cv2
from PIL import Image
import random
data_path='../data/First_round_data/jinnan2_round1_train_20190305'

############################################################
#  Dataset
############################################################

class JinNan(object):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    """参数:category决定数据类别为train validation test"""
    class_names_=['normal','restricted']
    def __init__(self, data_path='../data/First_round_data/jinnan2_round1_train_20190305'):
        
        self.data_path=data_path
        self._images_pathes=[]
        self._labels = []
        self.class_names_=['restricted','normal']
        self.image_ids=[]

    def load_JinNan(self,val=False,test=False):
        """
            加载图片信息:image path and label of each image
        """
        self.test=test
        if(not test):
            restricted_dir=os.path.join(data_path,'restricted/')
            normal_dir=os.path.join(data_path,'normal/')
            restricted_images=os.listdir(restricted_dir)
            normal_images=os.listdir(normal_dir)

            for file_name in restricted_images:
                # bg_color, shapes = self.random_image(height, width)
                im_path = os.path.join(restricted_dir, file_name)
                self.add_image(path=im_path,label=0) # 添加我的数据

            for file_name in normal_images:
                im_path = os.path.join(normal_dir, file_name)
                self.add_image(path=im_path,label=1) # 添加我的数据

            random.seed(32)
            random.shuffle(self._images_pathes)
            random.seed(32)
            random.shuffle(self._labels)
            
            if val:
                self._images_pathes=self._images_pathes[3000:]
                self._labels=self._labels[3000:]
            else:
                self._images_pathes=self._images_pathes[:3000]
                self._labels=self._labels[:3000]

            for x in range(len(self._labels)):
                self.image_ids.append(x)
        else:
            test_dataset_dir=self.data_path
            test_image_names=os.listdir(test_dataset_dir)
            test_image_names.sort(key=lambda x:int(x[:-4]))
            for id,image_name in enumerate(test_image_names):
                im_path=os.path.join(self.data_path,image_name)
                self._images_pathes.append(im_path)
                self.image_ids.append(id)


    def add_image(self,path,label):
        self._images_pathes.append(path)
        self._labels.append(label)

    def load_images(self,id):
        image = Image.open(self._images_pathes[id])
        image = image.resize((224, 224),Image.ANTIALIAS) 
        image = np.array(image)
        return image

    def load_label(self,id):
        return self._labels[id]

    def get_image_category(id):
        return self.class_names_[self._labels[id]]
    
    def __len__(self):
        return len(self.image_ids)

    def get_image_path(self,id):
        return self._images_pathes[id]


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from generator import data_generator
    from config import Config
    data_path='../data/First_round_data/jinnan2_round1_train_20190305'

    cfg=Config()
    
    dataset_tra=JinNan(data_path)
    dataset_tra.load_JinNan(val=False)
    dataset_tra_gen=data_generator(dataset_tra,cfg)

    dataset_val=JinNan(data_path)
    dataset_val.load_JinNan(val=True)
    dataset_tra_gen=data_generator(dataset_tra,cfg)
    

   
    print(len(dataset_tra))
    print(len(dataset_val))
    count=0
    for i in range(len(dataset_val)):
        #try:
            label=dataset_val.load_label(i)
            count+=label
            print(count)
        # except:
        #     print(i)

    
    