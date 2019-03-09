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
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

data_path='../data/First_round_data/jinnan2_round1_train_20190305'
model=inception_V3(2)

cfg=Config()

dataset_tra=JinNan(data_path)  
dataset_tra.load_JinNan(val=False)
dataset_tra_gen=data_generator(dataset_tra,cfg)

dataset_val=JinNan(data_path)
dataset_val.load_JinNan(val=True)
dataset_val_gen=data_generator(dataset_tra,cfg)


trainer=Trainer(model=model,mode='training',config=cfg,model_dir='./logs')

#trainer.load_weights(trainer.find_last(),by_name=True)
trainer.train(dataset_tra_gen,dataset_val_gen,0.001,100) 
 