# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------
import datetime
import os
import keras 
import logging
import multiprocessing
import numpy as np
import re
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

class Trainer(object):
    """Encapsulates the DeFCN model functionality.
    the actual Keras model is in the keras_model properity.
    """
    def __init__(self,model,mode,config,model_dir):
        """
        :param mode: Either "training" or "inference"
        :param config:  A Sub-class of the Config class
        :param model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training','inference']
        self.mode=mode
        self.config=config
        self.model_dir=model_dir
        self.set_log_dir()
        self.keras_model = self.build(model)

    def  build(self,model):
        """Build model's architecture."""
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
               model directory.
        Returns:
            he path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]

        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        if self.mode=='training':
            dir_name = os.path.join(self.model_dir, dir_names[-2])
            print(os.path.join(self.model_dir, dir_names[-1]))
            os.rmdir(os.path.join(self.model_dir, dir_names[-1]))
        else:
            dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.config.NAME.lower()), checkpoints)
        checkpoints = sorted(checkpoints)
        print(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self,filepath,by_name=False,exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weight(self):
        """Downloads ImageNet  trained weights form Keras.
        Return path to weights file."""
        #return weights_path
    def compile(self,learning_rate,momentum):
        """Gets the model ready for training.Adds losses,regulatization,and 
        metrics.Then call the Kerass compile() function"""
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,nesterov=True)

        self.keras_model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])


    def set_trainable(self,layer_regex,keras_modle=None,indent=0,verbose=1):
        """Sets model layres as trainable if their names match  the given regualr expression"""
    def set_log_dir(self,model_path=None):
        """Set the model log directory and epoch counter.
        model_path:If None ,or a format different form what this code uses then set a new 
        log directory and start epochs from 0. Otherwise,extract  the log directory and 
        the epoch counter form the file name.
        """
        if self.mode=='training':
            self.epoch=0
            now=datetime.datetime.now()
            #if we hanbe a model path with date and epochs use them
            if model_path:
                # Continue form we left of .Get epoch and date form the file name
                # A sample model path might look like:
                #/path/to/logs/coco2017.../DeFCN_0001.h5
                regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/[\w-]+(\d{4})\.h5"
                m = re.match(regex,model_path)
                if m:
                    now=datetime.datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),
                                          int(m.group(4)),int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    self.epoch = int(m.group(6)) - 1 + 1
                    print('Re-starting from epoch %d' % self.epoch)

                    # Directory for training logs
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                    self.config.NAME.lower(), now))
                # Create log_dir if not exists
            if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

                # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
                    self.config.NAME.lower()))
            self.checkpoint_path = self.checkpoint_path.replace(
                    "*epoch*", "{epoch:04d}")
    def train(self,train_dataset,val_datset,learning_rate,epochs,augmentation=None):
        """Train the model.
                train_dataset, val_dataset: Training and validation Dataset objects.
                learning_rate: The learning rate to train with
                epochs: Number of training epochs. Note that previous training epochs
                        are considered to be done alreay, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
                layers: Allows selecting wich layers to train. It can be:
                    - A regular expression to match layer names to train
                    - One of these predefined values:
                      heads: The RPN, classifier and mask heads of the network
                      all: All the layers
                      3+: Train Resnet stage 3 and up
                      4+: Train Resnet stage 4 and up
                      5+: Train Resnet stage 5 and up
                augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
                    augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
                    flips images right/left 50% of the time. You can pass complex
                    augmentations as well. This augmentation applies 50% of the
                    time, and when it does it flips images right/left half the time
                    and adds a Gausssian blur with a random sigma in range 0 to 5.
                        augmentation = imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ])
        """
        assert self.mode == "training", "Create model in training mode."
        train_generator = train_dataset
        val_generator = val_datset

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        # TODO:set trainable layrers
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        
        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )

        self.epoch = max(self.epoch, epochs)
    def detect(self,images,verbose=0):
        """Runs the detection pipeline.
                images: List of images, potentially of different sizes.
                Returns  a mask of image.
        """
        assert self.mode == "inference", "Create model in inference mode."
        images=np.reshape(images,[-1,224,224,3])
        result=self.keras_model.predict(images,batch_size=1)
        return result