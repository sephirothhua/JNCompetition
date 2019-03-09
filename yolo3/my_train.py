"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import cv2,os
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import copy
import time
from keras.optimizers import Adam

READ_NAGETIVE = True


def _main():
    annotation_path = './data/train_r1.txt'
    nagetive_path = './data/train_normal_r1.txt'
    log_dir = 'logs/001/'
    classes_path = 'model_data/danger.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (640,640) # multiple of 32, hw
    model = create_model(input_shape, anchors, len(class_names),load_pretrained=True,weights_path="logs/test_model/test_model_0309.h5")
    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir,negative=READ_NAGETIVE,nagetive_path=nagetive_path)

def train(model, annotation_path, input_shape, anchors, num_classes, log_dir='logs/',negative=False,nagetive_path=None):
    # model.compile(optimizer=Adam(lr=1e-3), loss={
    #     'yolo_loss': lambda y_true, y_pred: y_pred})
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    batch_size = 2
    val_split = 0.1
    if not negative:
        with open(annotation_path) as f:
            lines = f.readlines()
    else:
        with open(annotation_path) as f:
            positive_lines = f.readlines()
        with open(nagetive_path) as f:
            negative_lines = f.readlines()

    if not negative:
        np.random.seed(10)
        np.random.shuffle(lines)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    else:
        np.random.seed(10)
        np.random.shuffle(positive_lines)
        np.random.shuffle(negative_lines)
        p_num_val = int(len(positive_lines)*val_split)
        n_num_val = int(len(negative_lines)*val_split)
        p_num_train = len(positive_lines) - p_num_val
        n_num_train = len(negative_lines) - n_num_val
        print('Train on {} positive samples,{} nagetive samples, val on {} positive samples,{} nagetive samples, with batch size {}.'.format(p_num_train,n_num_train, p_num_val,n_num_val, batch_size))

    if not negative:
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=0,
                callbacks=[logging,checkpoint])
        model.save_weights(log_dir + 'trained_weights.h5')
    else:
        model.fit_generator(data_nagetive_generator_wrap(positive_lines[:p_num_train],negative_lines[:n_num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, (p_num_train+n_num_train)//batch_size),
                validation_data=data_nagetive_generator_wrap(positive_lines[p_num_train:],negative_lines[n_num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, (p_num_val+n_num_val)//batch_size),
                epochs=100,
                initial_epoch=0,
                callbacks=[logging,checkpoint])
        model.save_weights(log_dir + 'trained_weights.h5')

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False,
            weights_path='model_data/yolo_weights.h5'):
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model

def save_image(image,box):
    for i in range(len(box)):
        x1, y1, x2, y2, cls = box[i]
        cv2.rectangle((image*255).astype(np.uint8), (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print("%s" % (cls))
        cv2.putText(image, "{}".format(cls), (x1, y1 - 1), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 255, 255), 2)
        cv2.imwrite(os.path.join("./result", str(int(time.time()))), image)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            # save_image(image,box)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_negative_generator(positive_lines,nagetive_lines, batch_size, input_shape, anchors, num_classes):
    p = len(positive_lines)
    np.random.shuffle(positive_lines)
    p_i = 0
    n = len(nagetive_lines)
    np.random.shuffle(nagetive_lines)
    n_i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if np.random.random()<.5:
                p_i %= p
                image, box = get_random_data(positive_lines[p_i], input_shape, random=True)
                # save_image(image,box)
                image_data.append(image)
                box_data.append(box)
                p_i += 1
            else:
                n_i %= n
                image, box = get_random_data(nagetive_lines[n_i], input_shape, random=True)
                # save_image(image,box)
                image_data.append(image)
                box_data.append(box)
                n_i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def data_nagetive_generator_wrap(positive_lines,negative_lines, batch_size, input_shape, anchors, num_classes):
    p = len(positive_lines)
    n = len(negative_lines)
    if n==0 or p==0 or batch_size<=0: return None
    return data_negative_generator(positive_lines,negative_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()