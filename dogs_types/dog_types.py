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

import cv2
import numpy as np
import tensorflow as tf

from IPython.core.display import SVG
from keras.applications.vgg16 import VGG16
from keras.engine.sequential import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import model_to_dot, plot_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

OBJECTS_SIZE = 7000

model_name = "vgg16_dogs_2.dump"

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

LABEL_SIZE = 0

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
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
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

    def define_label(parent_name):
        return "-".join(parent_name.split('-')[1:])

    for subdir, dirs, files in os.walk(image_dir):
        print(f'PATH: {subdir} is processing')
        count = 0
        for file in files:
            path = pathlib.Path(subdir).absolute() / file
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image = cv2.resize(image, WH)
            image_label = define_label(path.parent.name)

            TRAIN_X.append(np.array(image))
            TRAIN_Y.append(image_label)
            count += 1
            print(f'Executed {count} / 120')

    if dump:
        file_x = open(DATA_DIR / 'xes.dump', 'wb')
        file_y = open(DATA_DIR / 'ykes.dump', 'wb')

        pickle.dump(TRAIN_X, file_x)
        pickle.dump(TRAIN_Y, file_y)
        file_x.close()
        file_y.close()


def get_dumped_data(**kwargs):
    global TRAIN_X, TRAIN_Y

    file_x = open(DATA_DIR / 'xes.dump', 'rb')
    file_y = open(DATA_DIR / 'ykes.dump', 'rb')

    TRAIN_X = pickle.load(file_x)
    TRAIN_Y = pickle.load(file_y)
    file_x.close()
    file_y.close()


def prepare_data(**kwargs):
    """ Do some normalizations and encode labels
    """
    global TRAIN_Y, TRAIN_X, LABEL_SIZE, x_train, y_train, x_test, y_test
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(TRAIN_Y[:OBJECTS_SIZE])

    LABEL_SIZE = len(set(Y))

    del TRAIN_Y
    # convert to binary matrix (120 labels at all) 2^10 = 128
    Y = to_categorical(Y[:OBJECTS_SIZE])
    X = np.array(TRAIN_X[:OBJECTS_SIZE])
    del TRAIN_X
    # normalize image
    X = X / 255
    # split image
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=69)

    with open('x_test.dump', 'wb') as file:
        pickle.dump(x_test, file)
    with open('y_test.dump', 'wb') as file:
        pickle.dump(y_test, file)


def augmentation(**kwargs):
    global x_train
    augs_gen.fit(x_train)
    return Y


def create_model(**kwargs):
    base_model = VGG16(include_top=False,
                       input_shape=(WIDHT, HEIGHT, 3),
                       weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers:
        print(layer, layer.trainable)

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(LABEL_SIZE, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def fit(**kwargs):
    model = kwargs['returnable']

    history = model.fit_generator(
        augs_gen.flow(x_train, y_train, batch_size=16),
        validation_data=(x_test, y_test),
        validation_steps=1000,
        steps_per_epoch=1000,
        epochs=15,
        verbose=1,
    )
    model.save(model_name)
    return history


if __name__ == '__main__':
    pipe = [
        #  read_dataset,
        get_dumped_data,
        prepare_data,
        augmentation,
        create_model,
        fit,
    ]
    prev_step_value = None
    for step in pipe:
        next_step_value = step(returnable=prev_step_value)
        prev_step_value = next_step_value

    #  file_name = 'gshep.jpg'

    #  image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    #  image = cv2.resize(image, WH)
    #  X = np.array(image) / 255
    #  X = augs_gen.apply_transform(X, {})
    #  X = np.expand_dims(X, axis=0)
    #  model = load_model(model_name)
    #  results = model.predict([X, ])

    #  indice = np.argmax(results[0], axis=0)


    #  file_y = open(DATA_DIR / 'ykes.dump', 'rb')
    #  TRAIN_Y = pickle.load(file_y)
    #  file_y.close()

    #  print(f"Labels: {set(TRAIN_Y)}")

    #  label_encoder = LabelEncoder()
    #  Y = label_encoder.fit_transform(TRAIN_Y[:OBJECTS_SIZE])
    #  print(label_encoder.inverse_transform([indice, ]))

    #  for idx in results.argsort()[0][::-1][:5]:
    #      print("{:.2f}%".format(results[0][idx]*100), "\t", label_encoder.inverse_transform([idx, ]))

