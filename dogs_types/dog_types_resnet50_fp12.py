"""
Tensorflow: 1.14
Python: 3.7

Classify dog types
"""
import logging
import os
import pathlib
import pickle
import xml.etree.ElementTree as ET
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from IPython.core.display import SVG
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.engine.sequential import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import model_to_dot, plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras.backend as K

dtype = 'float32'
K.set_floatx(dtype)
K.set_epsilon(1e-4)


OBJECTS_SIZE = 7000

model_name = "resnet50_dogs_initial3"

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'

DATASET_DIR = DATA_DIR / 'stanford-dogs-dataset'

ANNOTATIONS_DIR = DATASET_DIR / 'Annotation'
IMAGE_DIR = DATASET_DIR / 'Images'


WIDHT = 150
HEIGHT = 150
WH = (WIDHT, HEIGHT)

X = []
Y = []

LABEL_SIZE = 120

# rough
TRAIN_X = []
TRAIN_Y = []

# splitted data
x_test = []
y_test = []
x_train = []
y_train = []

# augment our image dataset
augs_gen = ImageDataGenerator(
    #  featurewise_center=False,
    #  samplewise_center=False,
    #  featurewise_std_normalization=False,
    #  samplewise_std_normalization=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='nearest',
    #  rescale=1.0 / 255.0,
)


def get_average_image_size(annotation_dir: str = ANNOTATIONS_DIR, **kwargs):
    """ 500 x 375 mid value
    Helper function
    """
    widths = []
    heights = []
    for subdir, dirs, files in os.walk(annotation_dir):
        for file in files:
            tree = ET.parse(annotation_dir / subdir / file)
            size = tree.getroot().find('size')
            widths.append(int(size.find('width').text))
            heights.append(int(size.find('height').text))

    def get_mid(lst): return sorted(lst)[int(len(lst) / 2)]
    return get_mid(widths), get_mid(heights)


def read_dataset(image_dir: str = IMAGE_DIR, dump: bool = True, **kwargs):
    """ Read and resize images
    Save all the data in:
        TRAIN_X - pixels
        TRAIN_Y - labels
    """
    global TRAIN_X, TRAIN_Y
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=(WIDHT, HEIGHT, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    #  model.add(Dense(512, activation='relu'))
    model.add(Dense(LABEL_SIZE, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    def define_label(parent_name):
        return "-".join(parent_name.split('-')[1:])

    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            path = pathlib.Path(subdir).absolute() / file
            image_label = define_label(path.parent.name)
            TRAIN_Y.append(image_label)

    label_encoder = LabelEncoder()
    TRAIN_Y = label_encoder.fit_transform(TRAIN_Y)
    TRAIN_Y = np.array(to_categorical(TRAIN_Y, num_classes=LABEL_SIZE))

    count = 0
    current_length_train_x = 0

    for subdir, dirs, files in os.walk(image_dir):
        print(f'PATH: {subdir} is processing')
        count += 1
        for file in files:
            path = pathlib.Path(subdir).absolute() / file
            image = load_img(str(path), target_size=WH)
            TRAIN_X.append(np.array(image))

        if count % 40 == 0:
            slice_left = current_length_train_x
            slice_right = slice_left + len(TRAIN_X)
            current_length_train_x = slice_right
            # convert to binary matrix (120 labels at all) 2^10 = 128
            # normalize image
            # split image

            # TODO: make active on resume iterations
            #  if count == 40:
            #      # make empty
            #      TRAIN_X = []
            #      model = load_model(f'{model_name}_iter_40.dump')
            #      continue

            x_train, x_test, y_train, y_test = train_test_split(
                np.array(TRAIN_X),
                TRAIN_Y[slice_left:slice_right],
                test_size=0.2,
                random_state=69,
            )

            # make empty
            TRAIN_X = []

            augs_gen.fit(x_train)
            model.fit_generator(
                augs_gen.flow(x_train, y_train, batch_size=25),
                validation_data=(x_test, y_test),
                validation_steps=1000,
                steps_per_epoch=1000,
                epochs=20,
                verbose=1,
                callbacks=[tensorboard_callback],
            )
            del x_train, x_test, y_train, y_test
            model.save(f'{model_name}_iter_{count}.dump')

        print(f'Executed {count} / 121')
    print('Prepare to write data on the disk')
    #  if dump:
    #      with open(DATA_DIR / 'xes.dump', 'wb') as file_x:
    #          pickle.dump(TRAIN_X, file_x)
    #      with open(DATA_DIR / 'ykes.dump', 'wb') as file_y:
    #          pickle.dump(TRAIN_Y, file_y)

    #  print('Dumped on the disk')
    #  time.sleep(5)


#  def get_dumped_data(**kwargs):
#      """ Get dumped python-arrays features
#      """
#      global TRAIN_X, TRAIN_Y
#      with open(DATA_DIR / 'xes.dump', 'rb') as file:
#          TRAIN_X = pickle.load(file)
#      with open(DATA_DIR / 'ykes.dump', 'rb') as file:
#          TRAIN_Y = pickle.load(file)

#      print('Get dumped data')
#      time.sleep(5)


#  def prepare_data(**kwargs):
#      """ Do some normalizations and encode labels
#      By object size
#      """
#      global TRAIN_Y, TRAIN_X, LABEL_SIZE, x_train, y_train, x_test, y_test
#      #  # Dump numpy arrays data
#      #  with open(DATA_DIR / 'x_test.dump', 'wb') as file:
#      #      pickle.dump(x_test, file)
#      #  with open(DATA_DIR / 'y_test.dump', 'wb') as file:
#      #      pickle.dump(y_test, file)


#  def read_dumped_train_test_data():
#      global x_train, x_test, y_train, y_test
#      with open(DATA_DIR / 'x_test.dump', 'rb') as file:
#          x_test = pickle.load(file)
#      with open(DATA_DIR / 'y_test.dump', 'rb') as file:
#          y_test = pickle.load(file)


#  def augmentation(**kwargs):
#      global x_train
#      return Y


#  def create_model(**kwargs):
#      return model


#  def fit(**kwargs):
#      model = kwargs['returnable']


if __name__ == '__main__':
    pipe = [
        read_dataset,
        #  get_dumped_data,
        #  prepare_data,
        #  augmentation,
        #  create_model,
        #  fit,
    ]
    prev_step_value = None
    for step in pipe:
        next_step_value = step(returnable=prev_step_value)
        prev_step_value = next_step_value

    #  file_name = 'dob.jpg'

    #  x = np.array(load_img(file_name, target_size=(150, 150)))
    #  x = x.reshape((1, ) + x.shape)

    #  i = 0
    #  for batch in augs_gen.flow(x, batch_size=1, save_to_dir='preview',
    #                             save_prefix='dob', save_format='jpg'):
    #      i += 1
    #      if i > 20:
    #          break
    #  raise Exception('stop')

    #  x = augs_gen.apply_transform(x, {})
    #  x = np.expand_dims(x, axis=0)
    #  model = load_model(model_name)

    #  augs_gen.fit()

    #  results = model.predict_generator([x, ])

    #  indice = np.argmax(results[0], axis=0)

    #  with open(data_dir / 'ykes.dump', 'rb') as file_y:
    #      train_y = pickle.load(file_y)

    #  print(f"labels: {set(train_y)}")

    #  label_encoder = labelencoder()
    #  y = label_encoder.fit_transform(train_y[:objects_size])
    #  print(label_encoder.inverse_transform([indice, ]))

    #  for idx in results.argsort()[0][::-1][:5]:
    #      print("{:.2f}%".format(results[0][idx] * 100), "\t",
    #            label_encoder.inverse_transform([idx, ]))
