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


WIDHT = 200
HEIGHT = 200
WH = (WIDHT, HEIGHT)
BATCH_SIZE = 16
EPOCHS = 20

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
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1.0 / 255.0,
    validation_split=0.2,
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
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(LABEL_SIZE, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

    #  def define_label(parent_name):
    #      return "-".join(parent_name.split('-')[1:])

    #  count = 0
    #  for subdir, dirs, files in os.walk(image_dir):
    #      print(f'PATH: {subdir} is processing')
    #      count += 1
    #      for file in files:
    #          path = pathlib.Path(subdir).absolute() / file
    #          image = load_img(str(path), target_size=WH)
    #          TRAIN_X.append(np.array(image))

    #          image_label = define_label(path.parent.name)
    #          TRAIN_Y.append(image_label)

    #  label_encoder = LabelEncoder()
    #  TRAIN_Y = label_encoder.fit_transform(TRAIN_Y)
    #  TRAIN_Y = np.array(to_categorical(TRAIN_Y, num_classes=LABEL_SIZE))

    #  x_train, x_test, y_train, y_test = train_test_split(
    #      np.array(TRAIN_X),
    #      TRAIN_Y,
    #      test_size=0.2,
    #      random_state=69,
    #  )

    train_generator = augs_gen.flow_from_directory(
        directory=IMAGE_DIR,
        target_size=WH,
        batch_size=BATCH_SIZE,
        seed=1,
        shuffle=True,
        subset='training',
    )
    test_generator = augs_gen.flow_from_directory(
        directory=IMAGE_DIR,
        target_size=WH,
        batch_size=BATCH_SIZE,
        seed=1,
        shuffle=True,
        subset='validation',
    )

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    with open(DATA_DIR / 'generator_labels.dump', 'wb') as file:
        pickle.dump(labels, file)

    model.fit_generator(
        train_generator,
        validation_data=test_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=test_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[tensorboard_callback],
    )
    print('Prepare to write data on the disk')

    model.save(f'{model_name}_without_iter.dump')

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


    #  labels = {}
    #  with open(DATA_DIR / 'generator_labels.dump', 'rb') as file:
    #      labels = pickle.load(file)

    #  print(labels)

    #  file_name = 'rot.JPG'

    #  x = np.array(load_img(file_name, target_size=(150, 150)))

    #  #  i = 0
    #  #  for batch in augs_gen.flow(x, batch_size=1, save_to_dir='preview',
    #  #                             save_prefix='dob', save_format='jpg'):
    #  #      i += 1
    #  #      if i > 20:
    #  #          break
    #  #  raise Exception('stop')

    #  model_name = f'{model_name}_without_iter.dump'
    #  x = np.expand_dims(x, axis=0)
    #  x = augs_gen.apply_transform(x, {})

    #  model = load_model(model_name)

    #  augs_gen.fit(x)

    #  results = model.predict_generator(augs_gen.flow(x, batch_size=25))


    #  print(results)

    #  predictions = [labels[k] for k in results.argsort()[0][::-1]]

    #  indice = np.argmax(results[0], axis=0)

    #  print(indice, predictions)







    #  with open(DATA_DIR / 'ykes.dump', 'rb') as file_y:
    #      train_y = pickle.load(file_y)

    #  print(f"labels: {set(train_y)}")

    #  label_encoder = LabelEncoder()
    #  y = label_encoder.fit_transform(train_y)
    #  print(label_encoder.inverse_transform([indice, ]))

    #  for idx in results.argsort()[0][::-1][:5]:
    #      print("{:.2f}%".format(results[0][idx] * 100), "\t",
    #            label_encoder.inverse_transform([idx, ]))
