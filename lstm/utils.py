import io
import os
import pathlib
import pickle
import re
import string
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pymystem3 import Mystem

mystem = Mystem()


def preprocess(text, force=False, read=False):
    if not read:
        if os.path.exists('data/sentences.dump') and not force:
            with open('data/sentences.dump', 'rb') as file:
                return pickle.load(file)

    text = re.sub(r'(–|\xa0–|\,|_|[0-9])+', '', text)
    text = text.lower()

    sentences = text.split('.')
    data = []

    for sentence in sentences:
        #  data.append(mystem.lemmatize(sentence.lower()))
        words = []
        for word in sentence.split(" "):
            words.append(word.strip(' \n' + string.punctuation).strip('\xa0'))
            #  data.append(sentence.split(" ")))
        data.append(words)

    sentences = []
    for sentence in data:
        words = [word for word in sentence
                 #  if word not in stopwords.words('russian')
                 if word != ' '
                 and word.strip() not in string.punctuation
                 and '\xa0' not in word.strip()
                 and 'a0' not in word.strip()
                 and ']–' not in word.strip()
                 and len(word) >= 3
                 ]
        sentences.append(" ".join(words))

    if force and not read:
        with open('data/sentences.dump', 'wb') as file:
            pickle.dump(sentences, file)
    return sentences


def flat(X):
    flat_x = []
    for x in X:
        flat_x += x
    return np.array(flat_x)


def create_dataset(X, time_steps=1):
    Xs, ys = [], []
    X = flat(X)
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(X[i + time_steps])
    return np.array(Xs), np.array(ys)


def get_callbacks():
    MODELS_DIR = pathlib.Path('models')
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model_checkpoint = ModelCheckpoint(
        str(MODELS_DIR /
            'weights-{epoch:02d}-{val_loss:.2f}.hdf5'),
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
    return [
        tensorboard_callback,
        model_checkpoint,
        reduce_lron,
    ]


def save_tokenizer(FILE_NAME, dump_name='models/tokenizer.dump'):
    text = read_text(FILE_NAME)
    text = preprocess(text, force=False)

    max_words = 10000
    tokenizer = Tokenizer(
        num_words=max_words,
        filters='"#$%&()*+-/:;<=>@[\]^_`{|}~'
    )
    tokenizer.fit_on_texts(text)

    with open(dump_name, 'wb') as file:
        pickle.dump(tokenizer, file)


def read_text(file_name):
    text = ""
    if isinstance(file_name, str):
        file_name = [file_name, ]
    for fname in file_name:
        with io.open(fname, 'r', encoding='cp1251', errors='strict') as file:
            text += file.read()
    return text


def save_labels(file_name, y, y_encoded):
    MAP = {}
    for Y, Y_encoded in zip(y, y_encoded):
        MAP[np.argmax(Y_encoded, 0)] = Y

    with open(file_name, 'wb') as file:
        pickle.dump(MAP, file)
