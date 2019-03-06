import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            # r_image.save("./result.jpg")
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here

    YOLO.get_defaults("model_path")
    YOLO.get_defaults("anchors_path")
    YOLO.get_defaults("classes_path")
    yolo = YOLO(model_path="./logs/000/ep024-loss46.727-val_loss42.751.h5",
                classes_path="./model_data/danger.txt")
    detect_img(yolo)
