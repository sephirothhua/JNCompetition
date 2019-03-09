# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------
import logging
import numpy as np


def load_image_label(dataset,image_id):
    image=dataset.load_images(image_id)
    label=dataset.load_label(image_id)
    return image,label

def data_generator(dataset,config,shuffle=True, augment=False):
    """A generator that returns images and corresponding target mask.
    dataset: The Dataset object to pick data from
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    batch_size: How many images to return in each call
    Returns a Python generator. Upon calling next() on it, the
    generator returns one list, inputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - mask: [batch,H,W].
    """
    b = 0  # batch item index
    image_index = -1
    
    image_ids = np.copy(dataset.image_ids)
    
    error_count = 0
    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)
            image_id = image_ids[image_index]
            #image_meta:image_id,image_shape,windows.active_class_ids
            image,label=load_image_label(dataset,image_id)
            # Init batch arrays
            if b == 0:
                batch_images = np.zeros((config.BATCH_SIZE ,)+ image.shape, dtype=np.float32)
                batch_label = np.zeros((config.BATCH_SIZE),dtype=np.float32)
            batch_images[b] = image
            batch_label[b] = label
            b += 1
            # Batch full?
            # input_image,input_labels
            if b >= config.BATCH_SIZE:
                batch_label=np.reshape(batch_label,[config.BATCH_SIZE,1])
                inputs = (batch_images,batch_label)
                yield inputs
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(dataset._images_pathes[image_id]))
            error_count += 1
            if error_count > 5:
                raise

if __name__ == '__main__':
    
    from data import JinNan
    from config import Config
    import matplotlib.pyplot as plt
    data_path='../data/First_round_data/jinnan2_round1_train_20190305'

    cfg=Config()

    dataset=JinNan(data_path)
    dataset.load_JinNan()
    image=dataset.load_images(1)

    train_gen=data_generator(dataset,cfg,shuffle=True, augment=False)
    for i,data in enumerate(train_gen):
        images,label=data
        print(images.shape)

        if i>=0:
            break