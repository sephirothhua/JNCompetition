# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

def inception_V3(num_classes):

    # create the base pre-trained model
    base_model = InceptionV3(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    # this is the model we will train
    return Model(inputs=base_model.input, outputs=predictions)



if __name__=='__main__':
    from data import JinNan
    from generator import data_generator
    from config import Config
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
  
    from keras.optimizers import SGD
    data_path='../data/First_round_data/jinnan2_round1_train_20190305'
    dataset=JinNan(data_path)
    dataset.load_JinNan()


    cfg=Config()
    data_gen=data_generator(dataset,cfg)

    ceptionV3=inception_V3(2)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(ceptionV3.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in ceptionV3.layers[:249]:
        layer.trainable = False
    for layer in ceptionV3.layers[249:]:
        layer.trainable = True
    ceptionV3.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    ceptionV3.fit_generator(data_gen,steps_per_epoch=1000,epochs=100,validation_data=data_gen,validation_steps=100,use_multiprocessing=True)
