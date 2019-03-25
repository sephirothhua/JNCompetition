import glob
from PIL import Image
import os
from yolo import YOLO


def detect_img(yolo,indir,outdir):
    for jpgfile in glob.glob(indir):
        img = Image.open(jpgfile)
        img = yolo.detect_image(img)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    yolo.close_session()


if __name__ == '__main__':
    yolo = YOLO(model_path="./logs/000/ep024-loss46.727-val_loss42.751.h5",
                classes_path="./model_data/danger.txt")
    detect_img(yolo,indir = "./data/train_r1/*.jpg",outdir = "./result")