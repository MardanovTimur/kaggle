"""
Tensorflow: 1.14
Python: 3.7

Classify dog types
"""
import logging
import pathlib
import pickle
from datetime import datetime

import keras.backend as K  # isort: skip
import numpy as np
import tensorflow as tf  # isort: skip
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.engine.sequential import Sequential
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import load_model
from keras.regularizers import l2
from keras_preprocessing.image import ImageDataGenerator

from lib import found_images, get_images

#  dtype = 'float32'
#  K.set_floatx(dtype)
#  K.set_epsilon(1e-4)

model_name = "inception_v3"

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

# Dir paths
ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

DATASET_DIR = DATA_DIR / 'stanford-dogs-dataset'

ANNOTATIONS_DIR = DATASET_DIR / 'Annotation'
IMAGE_DIR = DATASET_DIR / 'Images'
VALIDATE_DIR = ROOT_DIR / 'validated'
DUMPS_DIR = ROOT_DIR / 'dumps'

GENERATED_MAP_OF_LABELS_FILENAME = 'generator_labels.dump'


# Image params
WIDHT = 250
HEIGHT = 250
WH = (WIDHT, HEIGHT)

# model params
BATCH_SIZE = 16
EPOCHS = 20

LABEL_SIZE = 120


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


def fit(image_dir: str = IMAGE_DIR, dump: bool = True, **kwargs):
    """ Read and resize images
    Save all the data in:
        TRAIN_X - pixels
        TRAIN_Y - labels
    """
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model_checkpoint = ModelCheckpoint(
        str(MODELS_DIR /
            'weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5'),
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    reduce_lron = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto'
    )

    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=(WIDHT, HEIGHT, 3)
                             )
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dense(LABEL_SIZE, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

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
        callbacks=[tensorboard_callback, model_checkpoint, reduce_lron],
    )
    print('Prepare to write data on the disk')

    model.save(f'{model_name}.dump')


def predict(images: list,
            model_path: str,
            show_all_labels=False,
            labels_map_path: str = DUMPS_DIR / GENERATED_MAP_OF_LABELS_FILENAME,
            steps: int = 1,
            ):
    """ Simple predict some images
    """
    labels = {}
    with open(labels_map_path, 'rb') as file:
        labels = pickle.load(file)

    # some magic
    for k, v in labels.items():
        labels[k] = '_'.join(labels[k].split('-')[1:])

    if images:
        x = augs_gen.apply_transform(get_images(images, WH), {})
        augs_gen.fit(x)
    else:
        raise AttributeError("No images provided")

    test_generator = augs_gen.flow(x, shuffle=False, batch_size=BATCH_SIZE)

    model = load_model(str(model_path))

    results = model.predict_generator(test_generator, steps=steps)

    r_results = []

    for batch in range(int(len(results) / steps)):
        batch_results = results[batch * steps:(batch + 1) * steps]
        r_results.append(np.average(batch_results, axis=0))

    for result in r_results:
        #  places indices
        rargsort = result.argsort()[::-1][:5]
        for indice in rargsort:
            print(f'{labels[indice]}: {result[indice]}')
        print('\n' + '-' * 10)


def stage_1():
    images = list(found_images(VALIDATE_DIR / 'preview'))

    predict(images, model_path=MODELS_DIR / 'inception_v3.dump',
            show_all_labels=True, steps=1)
    predict(images, model_path=MODELS_DIR / 'eights-improvement-17-0.75.hdf5',
            show_all_labels=True, steps=1)


def stage_2():
    images = (
        'rott.jpg',
        'rot.JPG',
        'gshep.jpg',
        'tpoodle.jpg',
    )
    predict(images, model_path=MODELS_DIR / 'inception_v3.dump',
            show_all_labels=True, steps=1)

    predict(images, model_path=MODELS_DIR / 'eights-improvement-03-0.72.hdf5',
            show_all_labels=True, steps=1)


if __name__ == '__main__':
    #  aug_params = {
    #      'brightness_range': [0, 5],
    #  }
    #  augment_images_in_dir(VALIDATE_DIR,
    #                        VALIDATE_DIR / 'preview',
    #                        aug_params=aug_params)
    pass
