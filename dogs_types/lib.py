import logging
import os
import xml.etree.ElementTree as ET
from copy import copy

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__file__)
logger.setLevel('INFO')


def get_images(paths: list, WH):
    """Get resized images by <WH>"""
    try:
        return np.array([np.array(load_img(file_name, target_size=WH))
                         for file_name in paths])
    except Exception:
        logger.error('Cannot open some images. Check paths')


def found_images(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            yield subdir + '/' + file


def get_average_image_size(annotation_dir, **kwargs):
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


def augment_images_in_dir(dir: str,
                          save_dir: str,
                          WH: list = (250, 250),
                          format='.jpg',
                          gen_size: int = 2,
                          aug_params: dict = {}):
    """ Do augmentation for each image in directory

    <format> with dot firstly
    """
    image_data_gen = ImageDataGenerator(
        fill_mode='nearest',
        **aug_params,
    )
    flow_params = {
        'save_to_dir': save_dir,
        'save_prefix': 'aug_{}',
        'save_format': format[1:],
        'shuffle': False,
        'batch_size': 16,
    }
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            image = get_images(filter(lambda x: format in x,
                                      [subdir + '/' + file]))[0]
            save_prefix = flow_params['save_prefix'].format(file.split('.')[0])
            flow_param = copy(flow_params)
            flow_param['save_prefix'] = save_prefix
            gen = image_data_gen.flow(np.expand_dims(image, 0), **flow_param)
            i = 0
            while (i < gen_size):
                gen.next()
                i += 1
