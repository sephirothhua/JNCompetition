# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------

from data import JinNan
from generator import data_generator
from config import Config
from trainer import Trainer
from model import  inception_V3
import keras
import os
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

test_dataset_dir='../data/First_round_data/jinnan2_round1_test_a_20190306'
cfg=Config()

test_dataset=JinNan(data_path=test_dataset_dir)
test_dataset.load_JinNan(test=True)


model=inception_V3(2)
trainer=Trainer(model=model,mode='inference',config=cfg,model_dir='./logs')
trainer.load_weights(trainer.find_last(),by_name=True)

count=0
for id in range(len(test_dataset)):
    image=test_dataset.load_images(id)
    pre=trainer.detect(image)
    print(np.argmax(pre))
    count+=np.argmax(pre)
    #result=K.argmax(pre)
    #print(result)
print(count)
